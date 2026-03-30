# -*- coding: utf-8 -*-
"""
EKF Servis Karşılaştırması — Sadece G11 Uydusu
================================================
Bu script, EKF modelini 5 farklı servis (CODE, GFZ, JAXA, GRG, SHA) üzerinde
sadece G11 uydusu için çalıştırır ve aşağıdaki grafikleri üretir:

1. Orbit Comparison (Original vs Predicted) — Her servis için ayrı tek panel
2. Model Error (3D Position Error)          — Her servis için ayrı tek panel
3. LOO-CV Bar Chart                         — Tüm servislerin karşılaştırması (tek grafik)
"""

import os
import sys
import time as timer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Proje modüllerini import et
from config import CONFIG
from data_utils import parse_sp3
from models import predict_kalman_filter

# =============================================================================
# KONFİGÜRASYON
# =============================================================================

TARGET_PRN = 11          # Sadece G11
CONSTELLATION = "G"      # GPS

# Servis adları ve SP3 dosya yolları (modular_project klasörüne göre göreli)
SERVICES = {
    "CODE": r"..\precise orbit\precise\COD0MGXFIN_20200330000_01D_05M_ORB.SP3",
    "GFZ":  r"..\precise orbit\precise\GFZ0MGXRAP_20200330000_01D_05M_ORB.SP3",
    "JAXA": r"..\precise orbit\precise\JAX0MGXFIN_20200330000_01D_05M_ORB.SP3",
    "GRG":  r"..\precise orbit\precise\GRG0MGXFIN_20200330000_01D_15M_ORB.SP3",
    "SHA":  r"..\precise orbit\precise\SHA0MGXRAP_20200330000_01D_15M_ORB.SP3",
}

# Çıktı klasörü
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "EKF_G11")

# Renk paleti
AXIS_COLORS = {'X': '#e74c3c', 'Y': '#2ecc71', 'Z': '#3498db'}
EKF_COLOR = '#1abc9c'

# Servis bazlı bar chart renkleri
SERVICE_COLORS = {
    'CODE': '#e74c3c',
    'GFZ':  '#3498db',
    'JAXA': '#2ecc71',
    'GRG':  '#9b59b6',
    'SHA':  '#f39c12',
}


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def _interpolate_truth(p_times, epoch_times, true_xyz):
    """Gerçek veriyi tahmin zamanlarına interpole eder."""
    return np.array([np.interp(p_times, epoch_times, true_xyz[:, j]) for j in range(3)]).T


def cross_validate_loo(epoch_times, sat_xyz, sat_clock, config):
    """LOO-CV ile EKF'nin 3D RMSE performansını ölçer."""
    n_epochs = len(epoch_times)
    errors = np.zeros((n_epochs, 3))
    
    for i in range(n_epochs):
        mask = np.ones(n_epochs, dtype=bool)
        mask[i] = False
        
        train_times = epoch_times[mask]
        train_xyz = sat_xyz[mask]
        train_clk = sat_clock[mask]
        p_at = np.array([epoch_times[i]])
        
        try:
            p_xyz, _ = predict_kalman_filter(train_times, train_xyz, train_clk, p_at, config)
            errors[i] = p_xyz[0] - sat_xyz[i]
        except Exception:
            errors[i] = np.nan
    
    valid = ~np.any(np.isnan(errors), axis=1)
    if not np.any(valid):
        return float('inf')
    
    rmse = np.sqrt(np.mean(errors[valid]**2, axis=0))
    rmse_3d_m = np.sqrt(np.sum(rmse**2)) * 1000
    return rmse_3d_m


# =============================================================================
# GRAFİK FONKSİYONLARI
# =============================================================================

def plot_orbit_comparison(epoch_times, true_xyz, p_times, p_xyz, service_name, output_dir):
    """
    Tek panelde EKF Orbit Comparison çizer.
    Başlık: "Satellite G11 Orbit Comparison Original vs. Predicted EKF on {SERVICE}"
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    true_interp = _interpolate_truth(p_times, epoch_times, true_xyz)
    error_3d_m = np.sqrt(np.sum((p_xyz - true_interp) ** 2, axis=1)) * 1000
    rmse = np.sqrt(np.mean(error_3d_m ** 2))
    
    true_hours = epoch_times / 3600
    p_hours = p_times / 3600
    step = max(1, len(p_times) // 1500)
    
    axis_labels = ['X', 'Y', 'Z']
    for j, axis_name in enumerate(axis_labels):
        col = AXIS_COLORS[axis_name]
        # Orijinal veri (kesikli çizgi)
        ax.plot(true_hours, true_xyz[:, j], color=col, linestyle='--',
                linewidth=1.2, alpha=0.6)
        # EKF tahmini (düz çizgi)
        ax.plot(p_hours[::step], p_xyz[::step, j], color=col,
                linewidth=1.8, alpha=0.9)
    
    ax.set_title(f'Satellite G{TARGET_PRN:02d} Orbit Comparison Original vs. Predicted EKF on {service_name}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (Hours)', fontsize=12)
    ax.set_ylabel('Position (km)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.2, label='Original'),
        Line2D([0], [0], color='gray', linestyle='-', linewidth=1.8, label='Predicted (EKF)'),
        Line2D([0], [0], color=AXIS_COLORS['X'], linewidth=2, label='X'),
        Line2D([0], [0], color=AXIS_COLORS['Y'], linewidth=2, label='Y'),
        Line2D([0], [0], color=AXIS_COLORS['Z'], linewidth=2, label='Z'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9, ncol=2)
    
    # RMSE bilgisi
    ax.text(0.02, 0.98, f'3D RMSE: {rmse:.2f} m', transform=ax.transAxes,
            fontsize=11, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=EKF_COLOR, alpha=0.2))
    
    plt.tight_layout()
    
    orbit_dir = os.path.join(output_dir, "orbit_comparison")
    os.makedirs(orbit_dir, exist_ok=True)
    png_path = os.path.join(orbit_dir, f"satellite_G{TARGET_PRN:02d}_orbit_comparison_EKF_on_{service_name}.png")
    fig.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  [SAVED] {png_path}")
    plt.close(fig)
    
    return rmse


def plot_model_error(epoch_times, true_xyz, p_times, p_xyz, service_name, output_dir):
    """
    Tek panelde EKF 3D Position Error çizer.
    Başlık: "Satellite G11 EKF 3D Position Error on {SERVICE}"
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    true_interp = _interpolate_truth(p_times, epoch_times, true_xyz)
    error_3d_m = np.sqrt(np.sum((p_xyz - true_interp) ** 2, axis=1)) * 1000
    
    rmse = np.sqrt(np.mean(error_3d_m ** 2))
    p95 = np.percentile(error_3d_m, 95)
    max_err = np.max(error_3d_m)
    
    p_hours = p_times / 3600
    step = max(1, len(p_times) // 1500)
    
    ax.plot(p_hours[::step], error_3d_m[::step], color=EKF_COLOR, linewidth=1.8, alpha=0.85, label='3D Error')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3, label='Zero Error')
    
    # RMSE yatay çizgisi
    ax.axhline(rmse, color='#e74c3c', linestyle=':', alpha=0.6, linewidth=1.2, label=f'RMSE: {rmse:.2f} m')
    
    ax.set_title(f'Satellite G{TARGET_PRN:02d} EKF 3D Position Error on {service_name}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (Hours)', fontsize=12)
    ax.set_ylabel('3D Error (m)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # İstatistik kutusu
    stats_text = f'RMSE: {rmse:.2f} m\nMax: {max_err:.2f} m\n95th: {p95:.2f} m'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=EKF_COLOR, alpha=0.15))
    
    plt.tight_layout()
    
    error_dir = os.path.join(output_dir, "model_errors")
    os.makedirs(error_dir, exist_ok=True)
    png_path = os.path.join(error_dir, f"satellite_G{TARGET_PRN:02d}_model_error_EKF_on_{service_name}.png")
    fig.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  [SAVED] {png_path}")
    plt.close(fig)


def plot_loocv_comparison(cv_results, output_dir):
    """
    Tüm servislerin EKF LOO-CV 3D RMSE'sini karşılaştıran bar chart.
    """
    # Geçerli sonuçları filtrele
    valid_results = {k: v for k, v in cv_results.items() if np.isfinite(v)}
    if not valid_results:
        print("  [UYARI] Hiçbir serviste geçerli LOO-CV sonucu alınamadı!")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(valid_results.keys())
    values_m = [valid_results[n] for n in names]
    colors = [SERVICE_COLORS.get(n, 'gray') for n in names]
    
    bars = ax.bar(names, values_m, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
    
    ax.set_title(f'EKF LOO-CV 3D RMSE Comparison Across Services (G{TARGET_PRN:02d})',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('3D RMSE (m)', fontsize=12)
    ax.set_xlabel('Service', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Bar etiketleri
    for bar, val in zip(bars, values_m):
        if val >= 1000:
            label = f'{val/1000:.2f} km'
        else:
            label = f'{val:.2f} m'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                label, ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    cv_dir = os.path.join(output_dir, "loo_cv")
    os.makedirs(cv_dir, exist_ok=True)
    png_path = os.path.join(cv_dir, f"loo_cv_EKF_G{TARGET_PRN:02d}_services.png")
    fig.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  [SAVED] {png_path}")
    plt.close(fig)


# =============================================================================
# ANA İŞLEM
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print(f"  EKF SERVİS KARŞILAŞTIRMASI — Uydu G{TARGET_PRN:02d}")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cv_results = {}  # LOO-CV sonuçları (servis -> RMSE)
    
    for service_name, sp3_rel_path in SERVICES.items():
        print(f"\n{'─' * 60}")
        print(f"  Servis: {service_name}")
        print(f"{'─' * 60}")
        
        # 1. SP3 dosyasını oku
        sp3_path = os.path.abspath(os.path.join(base_dir, sp3_rel_path))
        if not os.path.exists(sp3_path):
            print(f"  [HATA] SP3 dosyası bulunamadı: {sp3_path}")
            continue
        
        e_times, s_data, header_interval = parse_sp3(sp3_path, CONSTELLATION)
        
        # Adım değerini güncelle
        config = CONFIG.copy()
        if header_interval is not None:
            config["predict_step_sec"] = int(header_interval)
        
        # 2. G11 verisini kontrol et
        if TARGET_PRN not in s_data:
            print(f"  [HATA] G{TARGET_PRN:02d} uydusu bu dosyada bulunamadı!")
            continue
        
        sat_xyz = s_data[TARGET_PRN][:, 0:3]
        sat_clk = s_data[TARGET_PRN][:, 3:4]
        
        # 3. Tahmin zamanlarını hazırla
        p_start = config["predict_start_sec"]
        p_end = int(e_times[-1])
        p_step = config["predict_step_sec"]
        p_times = np.arange(p_start, p_end, p_step, dtype=float)
        
        # 4. EKF çalıştır
        print(f"  EKF çalıştırılıyor...")
        start = timer.time()
        try:
            p_xyz, p_clk = predict_kalman_filter(e_times, sat_xyz, sat_clk, p_times, config)
            elapsed = timer.time() - start
            print(f"  EKF: OK ({elapsed:.2f}s)")
        except Exception as e:
            print(f"  EKF: FAIL — {e}")
            continue
        
        # 5. Orbit Comparison grafiği
        rmse = plot_orbit_comparison(e_times, sat_xyz, p_times, p_xyz, service_name, OUTPUT_DIR)
        print(f"  Orbit Comparison 3D RMSE: {rmse:.2f} m")
        
        # 6. Model Error grafiği
        plot_model_error(e_times, sat_xyz, p_times, p_xyz, service_name, OUTPUT_DIR)
        
        # 7. LOO-CV
        print(f"  LOO-CV hesaplanıyor (bu biraz sürebilir)...")
        cv_start = timer.time()
        try:
            cv_rmse = cross_validate_loo(e_times, sat_xyz, sat_clk, config)
            cv_elapsed = timer.time() - cv_start
            cv_results[service_name] = cv_rmse
            if np.isfinite(cv_rmse):
                print(f"  LOO-CV 3D RMSE: {cv_rmse:.3f} m ({cv_elapsed:.1f}s)")
            else:
                print(f"  LOO-CV: Geçerli sonuç alınamadı ({cv_elapsed:.1f}s)")
        except Exception as e:
            print(f"  LOO-CV: FAIL — {e}")
            cv_results[service_name] = float('inf')
    
    # 8. LOO-CV karşılaştırma bar chart
    print(f"\n{'─' * 60}")
    print(f"  LOO-CV Servis Karşılaştırma Grafiği")
    print(f"{'─' * 60}")
    plot_loocv_comparison(cv_results, OUTPUT_DIR)
    
    # Özet Tablo
    print(f"\n{'=' * 55}")
    print(f"  {'Servis':10s} | {'LOO-CV 3D RMSE (m)':>20s}")
    print(f"  {'─' * 50}")
    for name in sorted(cv_results, key=lambda k: cv_results[k]):
        val = cv_results[name]
        stat = f"{val:20.3f}" if np.isfinite(val) else "HATA"
        print(f"  {name:10s} | {stat}")
    print(f"  {'─' * 50}")
    
    print("\n" + "=" * 70)
    print("  İŞLEM BAŞARIYLA TAMAMLANDI!")
    print(f"  Çıktılar: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
