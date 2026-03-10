# -*- coding: utf-8 -*-
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import lagrange


def scale_and_predict(model, X_train, y_train, X_pred):
    """
    StandardScaler uygulayarak modeli eğiten ve tahmin yapan DRY yardımcı fonksiyonu.
    """
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_pred_scaled = scaler_x.transform(X_pred)

    model.fit(X_train_scaled, y_train_scaled.ravel())
    pred_scaled = model.predict(X_pred_scaled)

    return scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))


def _predict_clock_linear(epoch_times, sat_clock, predict_times):
    """
    Saat tahmini için LinearRegression yardımcı fonksiyonu.
    Uydu saatleri genelde lineer kayma (drift) gösterir, bu nedenle
    RF, Lagrange ve EKF gibi modellerde saat için LR tercih edilir.
    """
    lr = LinearRegression()
    lr.fit(epoch_times.reshape(-1, 1), sat_clock.ravel())
    return lr.predict(predict_times.reshape(-1, 1)).reshape(-1, 1)


def predict_random_forest(epoch_times, sat_xyz, sat_clock, predict_times, config):
    """
    Random Forest: XYZ için çoklu çıkış destekler (tek model).
    Saat için LR kullanılır çünkü RF extrapolasyon yapamaz, saat ise lineer kayma gösterir.
    """
    X_train = epoch_times.reshape(-1, 1)
    X_pred = predict_times.reshape(-1, 1)
    rf = RandomForestRegressor(n_estimators=config["rf_n_estimators"], random_state=config["rf_random_state"])
    rf.fit(X_train, sat_xyz)
    pred_xyz = rf.predict(X_pred)
    pred_clock = _predict_clock_linear(epoch_times, sat_clock, predict_times)
    return pred_xyz, pred_clock


def predict_svr(epoch_times, sat_xyz, sat_clock, predict_times, config):
    """
    SVR: Tek çıkışlı model, X/Y/Z/Saat her biri ayrı SVR ile tahmin edilir.
    """
    X_train = epoch_times.reshape(-1, 1)
    X_pred = predict_times.reshape(-1, 1)
    sat_all = np.hstack([sat_xyz, sat_clock])  # (n, 4) = X, Y, Z, Clock

    predictions = []
    for i in range(4):
        svr = SVR(kernel=config["svr_kernel"])
        predictions.append(scale_and_predict(svr, X_train, sat_all[:, i:i+1], X_pred))

    pred_xyz = np.hstack(predictions[:3])
    pred_clock = predictions[3]
    return pred_xyz, pred_clock


def predict_knn(epoch_times, sat_xyz, sat_clock, predict_times, config):
    """
    KNN: X/Y/Z/Saat her biri ayrı KNN ile tahmin edilir.
    Tutarlılık için scale_and_predict kullanılır.
    """
    X_train = epoch_times.reshape(-1, 1)
    X_pred = predict_times.reshape(-1, 1)
    sat_all = np.hstack([sat_xyz, sat_clock])  # (n, 4)

    predictions = []
    for i in range(4):
        knn = KNeighborsRegressor(n_neighbors=config["knn_n_neighbors"])
        predictions.append(scale_and_predict(knn, X_train, sat_all[:, i:i+1], X_pred))

    pred_xyz = np.hstack(predictions[:3])
    pred_clock = predictions[3]
    return pred_xyz, pred_clock


def predict_mlp(epoch_times, sat_xyz, sat_clock, predict_times, config):
    """
    MLP: X/Y/Z/Saat her biri ayrı MLP ile tahmin edilir.
    """
    X_train = epoch_times.reshape(-1, 1)
    X_pred = predict_times.reshape(-1, 1)
    sat_all = np.hstack([sat_xyz, sat_clock])  # (n, 4)

    predictions = []
    for i in range(4):
        mlp = MLPRegressor(hidden_layer_sizes=config["mlp_hidden_layers"],
                           max_iter=config["mlp_max_iter"], random_state=config["mlp_random_state"])
        predictions.append(scale_and_predict(mlp, X_train, sat_all[:, i:i+1], X_pred))

    pred_xyz = np.hstack(predictions[:3])
    pred_clock = predictions[3]
    return pred_xyz, pred_clock


def predict_lagrange(epoch_times, sat_xyz, sat_clock, predict_times, config):
    """
    Lagrange İnterpolasyonu: XYZ için pencereli polinom enterpolasyonu.
    Saat için LR kullanılır çünkü yüksek dereceli polinomlar kenarlarda
    aşırı salınım yapar (Runge fenomeni).
    StandardScaler kullanılmaz çünkü Lagrange bir ML algoritması değil,
    saf matematiksel bir eğri uydurma işlemidir; aksine veri ölçekleme
    polinom stabilitesini bozabilir.
    """
    n_epochs = len(epoch_times)
    w_half = config["lagrange_window"] // 2

    p_lists = [[], [], []]
    for t_val in predict_times:
        n_idx = np.argmin(np.abs(epoch_times - t_val))
        s_idx, e_idx = max(0, n_idx - w_half), min(n_epochs, n_idx + w_half)

        if e_idx - s_idx < config["lagrange_window"]:
            if s_idx == 0: e_idx = min(n_epochs, config["lagrange_window"])
            else: s_idx = max(0, n_epochs - config["lagrange_window"])

        w_t = epoch_times[s_idx:e_idx]
        w_t_shifted = w_t - w_t[0]
        t_val_shifted = t_val - w_t[0]

        for i in range(3):
            # sc_xyz yerine orijinal xyz değerlerini kullanıyoruz
            p_lists[i].append(lagrange(w_t_shifted, sat_xyz[s_idx:e_idx, i])(t_val_shifted))

    # sc_scalers yapısı kalktığı için doğrudan sonuçları alıyoruz
    pred_xyz = np.column_stack((p_lists[0], p_lists[1], p_lists[2]))
    pred_clock = _predict_clock_linear(epoch_times, sat_clock, predict_times)
    return pred_xyz, pred_clock


def predict_kalman_filter(epoch_times, sat_xyz, sat_clock, predict_times, config):
    """
    Gerçek Extended Kalman Filter (EKF) + RTS Smoother: 
    Yörünge dinamiği (İki Cisim Problemi) kullanılarak modellendi.
    Saat için LR kullanılır çünkü EKF'nin durum vektörüne saat eklemek
    tüm matris yapısının yeniden tasarlanmasını gerektirir.
    """
    MU = 398600.4418  # Dünya'nın standart kütleçekim parametresi (km^3/s^2)

    def orbit_dynamics(state):
        r = state[0:3]
        v = state[3:6]
        r_norm = np.linalg.norm(r)
        a = -MU / (r_norm**3) * r
        return np.hstack((v, a))

    def rk4_step(state, dt):
        if dt == 0:
            return state
        k1 = orbit_dynamics(state)
        k2 = orbit_dynamics(state + 0.5 * dt * k1)
        k3 = orbit_dynamics(state + 0.5 * dt * k2)
        k4 = orbit_dynamics(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def get_jacobian(state, dt):
        r = state[0:3]
        r_norm = np.linalg.norm(r)
        r5 = r_norm**5
        x, y, z = r
        
        G_r = np.zeros((3, 3))
        multiplier = -MU / (r_norm**3)
        
        G_r[0, 0] = multiplier + 3 * MU * x**2 / r5
        G_r[0, 1] = 3 * MU * x * y / r5
        G_r[0, 2] = 3 * MU * x * z / r5
        
        G_r[1, 0] = G_r[0, 1]
        G_r[1, 1] = multiplier + 3 * MU * y**2 / r5
        G_r[1, 2] = 3 * MU * y * z / r5
        
        G_r[2, 0] = G_r[0, 2]
        G_r[2, 1] = G_r[1, 2]
        G_r[2, 2] = multiplier + 3 * MU * z**2 / r5

        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 0:3] = G_r
        
        # Sürekli zaman Jacobian'ının Birinci Derece Taylor Seri Yaklaşımı
        # F = I + A*dt
        return np.eye(6) + A * dt

    n_epochs = len(epoch_times)
    n_states = 6
    H = np.zeros((3, n_states))
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0
    R = np.eye(3) * config["ekf_measurement_noise"]

    dt0 = epoch_times[1] - epoch_times[0]
    v0 = (sat_xyz[1] - sat_xyz[0]) / dt0
    x_curr = np.zeros(n_states)
    x_curr[0:3], x_curr[3:6] = sat_xyz[0], v0
    P_curr = np.diag([config["ekf_measurement_noise"]]*3 + [1e-2]*3)

    x_filt, P_filt, x_pred_list, P_pred_list = [], [], [], []
    for k in range(n_epochs):
        if k > 0:
            dt = epoch_times[k] - epoch_times[k-1]
            # RK4 ile non-lineer durum ilerletme
            x_pred_k = rk4_step(x_curr, dt)
            # Sistemin Jacobian matrisini hesaplama
            F = get_jacobian(x_curr, dt)
            Q = np.diag([config["ekf_process_noise_pos"]]*3 + [config["ekf_process_noise_vel"]]*3) * dt
            P_pred_k = F @ P_curr @ F.T + Q
        else:
            x_pred_k, P_pred_k = x_curr, P_curr

        S = H @ P_pred_k @ H.T + R
        K = P_pred_k @ H.T @ np.linalg.inv(S)
        x_curr = x_pred_k + K @ (sat_xyz[k] - H @ x_pred_k)
        P_curr = (np.eye(n_states) - K @ H) @ P_pred_k

        x_filt.append(x_curr)
        P_filt.append(P_curr)
        x_pred_list.append(x_pred_k)
        P_pred_list.append(P_pred_k)

    x_smooth = [np.zeros(n_states)] * n_epochs
    x_smooth[-1] = x_filt[-1]
    for k in range(n_epochs-2, -1, -1):
        dt = epoch_times[k+1] - epoch_times[k]
        F = get_jacobian(x_filt[k], dt)
        C = P_filt[k] @ F.T @ np.linalg.inv(P_pred_list[k+1])
        x_smooth[k] = x_filt[k] + C @ (x_smooth[k+1] - x_pred_list[k+1])

    pred_xyz = np.zeros((len(predict_times), 3))
    for i, t in enumerate(predict_times):
        # En yakın bilinen/düzleştirilmiş duruma göre dt'yi belirleyip non-lineer olarak ilerletmek
        idx = max(0, min(np.searchsorted(epoch_times, t) - 1, n_epochs - 1))
        dt = t - epoch_times[idx]
        x_sim = rk4_step(x_smooth[idx], dt)
        pred_xyz[i] = x_sim[0:3]

    pred_clock = _predict_clock_linear(epoch_times, sat_clock, predict_times)
    return pred_xyz, pred_clock
