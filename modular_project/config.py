# -*- coding: utf-8 -*-

# Proje genelinde kullanılan tüm sabitler ve model parametreleri burada yönetilir.

CONFIG = {
    # --- Veri Kaynakları ---
    "sp3_file": r"..\precise orbit\precise\SHA0MGXRAP_20200330000_01D_15M_ORB.SP3",
    "constellation": "G",                                # G=GPS, R=GLONASS, E=Galileo, C=BeiDou
    "satellite_range": (1, 32),                          # Analiz edilecek uydu PRN aralığı
    
    # --- Tahmin Zamanlaması ---
    "predict_start_sec": 1,                              # Tahminin başlayacağı saniye
    "predict_step_sec": 900,                             # Tahmin adımı (900s = 15 Dakika)
    
    # --- Makine Öğrenmesi Model Parametreleri ---
    "rf_n_estimators": 100,
    "rf_random_state": 0,
    "svr_kernel": "rbf",
    "knn_n_neighbors": 5,
    "mlp_max_iter": 1000,
    "mlp_random_state": 1,
    "mlp_hidden_layers": (50,),
    
    # --- Matematiksel Modeller ---
    "lagrange_window": 10,                              # Lagrange derecesi (10 nokta = 9. derece)
    
    # --- Kalman Filtresi (EKF) Gürültü Parametreleri ---
    "ekf_process_noise_pos": 1e-4,                      # Konum varyansı (km^2)
    "ekf_process_noise_vel": 1e-2,                      # Hız varyansı (km/s)^2
    "ekf_measurement_noise": 1e-6,                      # SP3 ölçüm gürültüsü
    
    # --- Çıktı Yönetimi ---
    "run_cross_validation": True,                       # Çapraz doğrulama çalıştırılsın mı?
}
