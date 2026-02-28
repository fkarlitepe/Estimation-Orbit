# -*- coding: utf-8 -*-
import os
import time as timer
import numpy as np
from typing import Dict, Any

# Diğer modüllerden fonksiyonları içe aktar
from config import CONFIG
from data_utils import parse_sp3, write_sp3
from models import (predict_random_forest, predict_svr, predict_knn, 
                    predict_mlp, predict_lagrange, predict_kalman_filter)
from metrics import compute_metrics, visualize_results


def cross_validate_leave_one_out(epoch_times, sat_xyz, sat_clock, method_func, config, method_name):
    """
    LOO-CV Çapraz doğrulama ile modellerin gerçek dünyada 'görmediği' verideki performansını ölçer.
    """
    n_epochs = len(epoch_times)
    errors = np.zeros((n_epochs, 3))
    
    for i in range(n_epochs):
        mask = np.ones(n_epochs, dtype=bool)
        mask[i] = False
        
        train_times, train_xyz, train_clk = epoch_times[mask], sat_xyz[mask], sat_clock[mask]
        p_at = np.array([epoch_times[i]])
        
        try:
            p_xyz, _ = method_func(train_times, train_xyz, train_clk, p_at, config)
            errors[i] = p_xyz[0] - sat_xyz[i]
        except:
            errors[i] = np.nan
            
    valid = ~np.any(np.isnan(errors), axis=1)
    if not np.any(valid):
        return 99999.0
        
    rmse = np.sqrt(np.mean(errors[valid]**2, axis=0))
    rmse_3d_m = np.sqrt(np.sum(rmse**2)) * 1000
    
    print(f"    {method_name:10s} LOO-CV 3D RMSE: {rmse_3d_m:.3f} Metre")
    return rmse_3d_m


def run_pipeline(config: Dict[str, Any]):
    """
    Ana işlem akışını yöneten koordinasyon fonksiyonu.
    """
    print("\n" + "="*70)
    print("  UYDU YÖRÜNGE TAHMİNİ - MODÜLER PROJE")
    print("="*70)

    # 1. Veri Okuma - Yolları dosya konumuna göre çöz
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sp3_path = config["sp3_file"]
    if not os.path.isabs(sp3_path):
        sp3_path = os.path.abspath(os.path.join(base_dir, sp3_path))
        
    if not os.path.exists(sp3_path):
        print(f"  [HATA] SP3 dosyası bulunamadı: {sp3_path}")
        return

    e_times, s_data, e_interval = parse_sp3(sp3_path, config["constellation"])
    
    # 2. Tahmin Zamanlarını Hazırlama
    p_start, p_end, p_step = config["predict_start_sec"], int(e_times[-1]), config["predict_step_sec"]
    p_times = np.arange(p_start, p_end, p_step, dtype=float)
    
    # 3. Modellerin Belirlenmesi
    methods = {
        'RF': predict_random_forest, 'SVR': predict_svr, 'KNN': predict_knn,
        'MLP': predict_mlp, 'Lagrange': predict_lagrange, 'EKF': predict_kalman_filter
    }
    
    prn_min, prn_max = config["satellite_range"]
    valid_prns = sorted([p for p in s_data if prn_min <= p <= prn_max])
    
    all_results = {m: {} for m in methods}
    out_dir = os.path.abspath(os.path.join(base_dir, config["output_dir"]))
    os.makedirs(out_dir, exist_ok=True)

    # 4. Modelleri Çalıştırma
    for idx, prn in enumerate(valid_prns):
        sat_xyz, sat_clk = s_data[prn][:, 0:3], s_data[prn][:, 3:4]
        print(f"\nSatellite PRN {prn:02d} ({idx+1}/{len(valid_prns)}) İşleniyor...")
        
        for name, func in methods.items():
            start = timer.time()
            try:
                p_xyz, p_clk = func(e_times, sat_xyz, sat_clk, p_times, config)
                all_results[name][prn] = (p_times, p_xyz, p_clk)
                print(f"    {name:10s}: OK ({timer.time()-start:.2f}s)")
            except Exception as e:
                print(f"    {name:10s}: FAIL - {e}")

    # 5. Görselleştirme (İlk Uydu İçin)
    if valid_prns:
        demo_prn = valid_prns[0]
        results_subset = {m: (v[demo_prn][0], v[demo_prn][1]) for m, v in all_results.items() if demo_prn in v}
        visualize_results(e_times, s_data[demo_prn][:, 0:3], results_subset, demo_prn, out_dir)

    # 6. Kayıt ve SP3 Çıktısı
    print(f"\nSonuçlar {out_dir} klasörüne kaydediliyor...")
    for name in methods:
        sat_res_sp3 = {}
        for p in valid_prns:
            if p in all_results[name]:
                sat_res_sp3[p] = (all_results[name][p][1], all_results[name][p][2])
        
        if sat_res_sp3:
            write_sp3(os.path.join(out_dir, f"est_{name}.sp3"), p_times, sat_res_sp3, p_step, config["constellation"])

    # 7. Çapraz Doğrulama (İsteğe Bağlı)
    if config.get("run_cross_validation"):
        print("\nÇapraz Doğrulama Testi Başlıyor (LOO-CV)...")
        cv_summary = {}
        target_prn = valid_prns[0]
        for name, func in methods.items():
            try:
                cv_summary[name] = cross_validate_leave_one_out(e_times, s_data[target_prn][:, 0:3], s_data[target_prn][:, 3:4], func, config, name)
            except:
                cv_summary[name] = 999999.0
        
        # Özet Tablosu Yazdır
        print("\n" + "-"*40)
        print(f"{'Yöntem':15s} | {'LOO-CV 3D RMSE (m)':>18s}")
        print("-"*40)
        for name in sorted(cv_summary, key=cv_summary.get):
            val = cv_summary[name]
            stat = f"{val:18.3f}" if val < 900000 else "HATA"
            print(f"{name:15s} | {stat}")
        print("-"*40)
        
    print("\n" + "="*70)
    print("  İŞLEM BAŞARIYLA TAMAMLANDI!")
    print("="*70)


if __name__ == "__main__":
    run_pipeline(CONFIG)
