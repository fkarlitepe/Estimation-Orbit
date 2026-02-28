# -*- coding: utf-8 -*-
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import lagrange
from typing import Tuple, Dict, Any


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


def predict_random_forest(epoch_times, sat_xyz, sat_clock, predict_times, config):
    X_train = epoch_times.reshape(-1, 1)
    X_pred = predict_times.reshape(-1, 1)
    rf = RandomForestRegressor(n_estimators=config["rf_n_estimators"], random_state=config["rf_random_state"])
    rf.fit(X_train, sat_xyz)
    pred_xyz = rf.predict(X_pred)
    
    rf_clk = RandomForestRegressor(n_estimators=config["rf_n_estimators"], random_state=config["rf_random_state"])
    rf_clk.fit(X_train, sat_clock.ravel())
    pred_clock = rf_clk.predict(X_pred).reshape(-1, 1)
    return pred_xyz, pred_clock


def predict_svr(epoch_times, sat_xyz, sat_clock, predict_times, config):
    X_train = epoch_times.reshape(-1, 1)
    X_pred = predict_times.reshape(-1, 1)
    
    p_xyz = []
    for i in range(3):
        svr = SVR(kernel=config["svr_kernel"])
        p_xyz.append(scale_and_predict(svr, X_train, sat_xyz[:, i:i+1], X_pred))
    
    pred_xyz = np.hstack(p_xyz)
    lr_clk = LinearRegression()
    lr_clk.fit(X_train, sat_clock.ravel())
    pred_clock = lr_clk.predict(X_pred).reshape(-1, 1)
    return pred_xyz, pred_clock


def predict_knn(epoch_times, sat_xyz, sat_clock, predict_times, config):
    X_train = epoch_times.reshape(-1, 1)
    X_pred = predict_times.reshape(-1, 1)
    
    p_xyz = []
    for i in range(3):
        knn = KNeighborsRegressor(n_neighbors=config["knn_n_neighbors"])
        knn.fit(X_train, sat_xyz[:, i])
        p_xyz.append(knn.predict(X_pred).reshape(-1, 1))
        
    pred_xyz = np.hstack(p_xyz)
    lr_clk = LinearRegression()
    lr_clk.fit(X_train, sat_clock.ravel())
    pred_clock = lr_clk.predict(X_pred).reshape(-1, 1)
    return pred_xyz, pred_clock


def predict_mlp(epoch_times, sat_xyz, sat_clock, predict_times, config):
    X_train = epoch_times.reshape(-1, 1)
    X_pred = predict_times.reshape(-1, 1)
    
    p_xyz = []
    for i in range(3):
        mlp = MLPRegressor(hidden_layer_sizes=config["mlp_hidden_layers"], 
                           max_iter=config["mlp_max_iter"], random_state=config["mlp_random_state"])
        p_xyz.append(scale_and_predict(mlp, X_train, sat_xyz[:, i:i+1], X_pred))
    
    pred_xyz = np.hstack(p_xyz)
    lr_clk = LinearRegression()
    lr_clk.fit(X_train, sat_clock.ravel())
    pred_clock = lr_clk.predict(X_pred).reshape(-1, 1)
    return pred_xyz, pred_clock


def predict_lagrange(epoch_times, sat_xyz, sat_clock, predict_times, config):
    n_epochs = len(epoch_times)
    w_half = config["lagrange_window"] // 2
    
    # Sayısal kararlılık için ölçeklendirme
    sc_xyz = [StandardScaler().fit_transform(sat_xyz[:, i:i+1]).ravel() for i in range(3)]
    sc_scalers = [StandardScaler().fit(sat_xyz[:, i:i+1]) for i in range(3)]

    p_lists = [[], [], []]
    for t_val in predict_times:
        # En yakın indeksi bul
        n_idx = np.argmin(np.abs(epoch_times - t_val))
        s_idx, e_idx = max(0, n_idx - w_half), min(n_epochs, n_idx + w_half)
        
        # Pencereyi doldur (Kelepçeleme)
        if e_idx - s_idx < config["lagrange_window"]:
            if s_idx == 0: e_idx = min(n_epochs, config["lagrange_window"])
            else: s_idx = max(0, n_epochs - config["lagrange_window"])
            
        w_t = epoch_times[s_idx:e_idx]
        for i in range(3):
            p_lists[i].append(lagrange(w_t, sc_xyz[i][s_idx:e_idx])(t_val))

    pred_xyz = np.hstack([sc_scalers[i].inverse_transform(np.array(p_lists[i]).reshape(-1, 1)) for i in range(3)])
    
    lr_clk = LinearRegression()
    lr_clk.fit(epoch_times.reshape(-1, 1), sat_clock.ravel())
    pred_clock = lr_clk.predict(predict_times.reshape(-1, 1)).reshape(-1, 1)
    return pred_xyz, pred_clock


# EKF ve RTS Smoother Bloğu
def predict_kalman_filter(epoch_times, sat_xyz, sat_clock, predict_times, config):
    n_epochs = len(epoch_times)
    n_states = 6
    H = np.zeros((3, n_states))
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0
    R = np.eye(3) * config["ekf_measurement_noise"]

    # Başlangıç Durumu
    dt0 = epoch_times[1] - epoch_times[0]
    v0 = (sat_xyz[1] - sat_xyz[0]) / dt0
    x_curr = np.zeros(n_states)
    x_curr[0:3], x_curr[3:6] = sat_xyz[0], v0
    P_curr = np.diag([config["ekf_measurement_noise"]]*3 + [1e-2]*3)

    # İleri Kalman (Forward Pass)
    x_filt, P_filt, x_pred_list, P_pred_list = [], [], [], []
    for k in range(n_epochs):
        if k > 0:
            dt = epoch_times[k] - epoch_times[k-1]
            F = np.eye(n_states)
            F[0, 3] = F[1, 4] = F[2, 5] = dt
            Q = np.diag([config["ekf_process_noise_pos"]]*3 + [config["ekf_process_noise_vel"]]*3) * dt
            x_pred_k, P_pred_k = F @ x_curr, F @ P_curr @ F.T + Q
        else:
            x_pred_k, P_pred_k = x_curr, P_curr
        
        S = H @ P_pred_k @ H.T + R
        K = P_pred_k @ H.T @ np.linalg.inv(S)
        x_curr = x_pred_k + K @ (sat_xyz[k] - H @ x_pred_k)
        P_curr = (np.eye(n_states) - K @ H) @ P_pred_k
        
        x_filt.append(x_curr); P_filt.append(P_curr)
        x_pred_list.append(x_pred_k); P_pred_list.append(P_pred_k)

    # RTS Geriye Düzeltme (Backward Smoother)
    x_smooth = [np.zeros(n_states)] * n_epochs
    x_smooth[-1] = x_filt[-1]
    for k in range(n_epochs-2, -1, -1):
        dt = epoch_times[k+1] - epoch_times[k]
        F = np.eye(n_states); F[0, 3] = F[1, 4] = F[2, 5] = dt
        C = P_filt[k] @ F.T @ np.linalg.inv(P_pred_list[k+1])
        x_smooth[k] = x_filt[k] + C @ (x_smooth[k+1] - x_pred_list[k+1])

    # Ara Saniyeleri Tahmin Etme
    pred_xyz = np.zeros((len(predict_times), 3))
    for i, t in enumerate(predict_times):
        idx = max(0, min(np.searchsorted(epoch_times, t) - 1, n_epochs - 1))
        dt = t - epoch_times[idx]
        F_dt = np.eye(n_states); F_dt[0, 3] = F_dt[1, 4] = F_dt[2, 5] = dt
        pred_xyz[i] = (F_dt @ x_smooth[idx])[0:3]

    lr_clk = LinearRegression()
    lr_clk.fit(epoch_times.reshape(-1, 1), sat_clock.ravel())
    pred_clock = lr_clk.predict(predict_times.reshape(-1, 1)).reshape(-1, 1)
    return pred_xyz, pred_clock
