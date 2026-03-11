# -*- coding: utf-8 -*-

# Proje genelinde kullanılan tüm sabitler ve model parametreleri burada yönetilir.

CONFIG = {
    # --- Veri Kaynakları ---
    # Not: Komut satırından `--sp3` argümanı ile ezilebilir.
    "sp3_file": r"..\precise orbit\precise\COD0MGXFIN_20200330000_01D_05M_ORB.SP3",
    # Not: Komut satırından `--cons` argümanı ile ezilebilir (G=GPS, R=GLONASS, E=Galileo, C=BeiDou).
    "constellation": "G",                                
    "satellite_range": (1, 32),                          # Analiz edilecek uydu PRN aralığı
    
    # --- Tahmin Zamanlaması ---
    "predict_start_sec": 1,                              # Tahminin başlayacağı saniye
    # Not: Ana kod içerisinde (main.py) parse_sp3'den gelen header_interval değeri ile otomatik ezilir.
    # Uyumsuzluk çıkmaması için bu değer artık sadece varsayılan bir yedeği temsil eder.
    "predict_step_sec": 900,                             # Tahmin adımı (Varsayılan 900s)
    
    # --- Makine Öğrenmesi Model Parametreleri ---
    "rf_n_estimators": 150,
    "rf_random_state": 0,
    "svr_kernel": "rbf",
    "knn_n_neighbors": 9,
    "mlp_max_iter": 2000,
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
