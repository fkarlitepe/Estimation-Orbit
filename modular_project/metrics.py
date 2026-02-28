# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Tuple


def compute_metrics(true_vals: np.ndarray, pred_vals: np.ndarray, label: str = "") -> Dict[str, np.ndarray]:
    """
    Tahmin edilen ve gerçek değerler arasındaki hatayı hesaplar.
    """
    diff = true_vals - pred_vals
    rmse = np.sqrt(np.mean(diff**2, axis=0))
    mae = np.mean(np.abs(diff), axis=0)
    max_err = np.max(np.abs(diff), axis=0)
    
    print(f"  {label:12s} | RMSE(km) (X={rmse[0]:.4f}, Y={rmse[1]:.4f}, Z={rmse[2]:.4f})")
    return {"rmse": rmse, "mae": mae, "max_error": max_err}


def visualize_results(epoch_times: np.ndarray, 
                      true_xyz: np.ndarray, 
                      results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                      sat_prn: int, 
                      output_dir: str) -> None:
    """
    Karşılaştırmalı performans grafiklerini PNG olarak kaydeder.
    """
    os.makedirs(output_dir, exist_ok=True)
    coord_labels = ['X Koordinatı (km)', 'Y Koordinatı (km)', 'Z Koordinatı (km)']
    colors = {'RF': '#e74c3c', 'SVR': '#3498db', 'KNN': '#2ecc71', 
              'MLP': '#9b59b6', 'Lagrange': '#f39c12', 'EKF': '#1abc9c'}

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Uydu G{sat_prn:02d} - Tahmin Yöntemleri Karşılaştırma Analizi', fontsize=14, fontweight='bold')

    for i in range(3):
        ax = axes[i]
        ax.scatter(epoch_times / 3600, true_xyz[:, i], color='black', s=20, zorder=5, label='Gerçek (SP3)')
        
        for name, (p_times, p_xyz) in results_dict.items():
            col = colors.get(name, 'gray')
            step = max(1, len(p_times) // 1500)
            ax.plot(p_times[::step] / 3600, p_xyz[::step, i], color=col, alpha=0.7, label=name)
        
        ax.set_ylabel(coord_labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(loc='upper right', fontsize=8)

    axes[2].set_xlabel('Zaman (Saat)')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"G{sat_prn:02d}_Analiz.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [GRAFİK] Kaydedildi: {plot_path}")
