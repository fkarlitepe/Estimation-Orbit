# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple


# Model renk haritası
MODEL_COLORS = {
    'RF': '#e74c3c', 'SVR': '#3498db', 'KNN': '#2ecc71',
    'MLP': '#9b59b6', 'Lagrange': '#f39c12', 'EKF': '#1abc9c'
}


def _interpolate_truth(p_times: np.ndarray, epoch_times: np.ndarray, true_xyz: np.ndarray) -> np.ndarray:
    """Gerçek veriyi tahmin zamanlarına interpole eder."""
    return np.array([np.interp(p_times, epoch_times, true_xyz[:, j]) for j in range(3)]).T


def visualize_model_grid(epoch_times: np.ndarray,
                         true_xyz: np.ndarray,
                         results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
                         sat_prn: int,
                         output_dir: str = "output") -> None:
    """
    Her modeli ayrı panelde gösterir (2x3 grid).
    Her panelde o modelin 3D pozisyon hatasını (metre) zaman bazlı çizer.
    """
    models = list(results_dict.keys())
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True)
    fig.suptitle(f'Satellite G{sat_prn:02d} - Model-Based 3D Position Error on SHA', fontsize=14, fontweight='bold')

    axes_flat = np.array(axes).flatten() if n_models > 1 else [axes]

    print(f"\n  {'Model':12s} | {'3D RMSE (m)':>12s} | {'Max Error (m)':>14s} | {'95th Percentile (m)':>18s}")
    print("  " + "-" * 65)

    for i, name in enumerate(models):
        ax = axes_flat[i]
        p_times, p_xyz = results_dict[name]
        col = MODEL_COLORS.get(name, 'gray')

        true_interp = _interpolate_truth(p_times, epoch_times, true_xyz)
        error_3d_m = np.sqrt(np.sum((p_xyz - true_interp) ** 2, axis=1)) * 1000

        step = max(1, len(p_times) // 1500)
        ax.plot(p_times[::step] / 3600, error_3d_m[::step], color=col, linewidth=1.5, alpha=0.85, label='3D Error')
        ax.axhline(0, color='black', linestyle='--', alpha=0.3, label='Zero Error')

        rmse = np.sqrt(np.mean(error_3d_m ** 2))
        p95 = np.percentile(error_3d_m, 95)
        max_err = np.max(error_3d_m)

        ax.set_title(f'{name}  (RMSE: {rmse:.2f} m)', fontsize=11, fontweight='bold', color=col)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('3D Error (m)')
        ax.legend(loc='upper right', fontsize=8)

        print(f"  {name:12s} | {rmse:12.3f} | {max_err:14.3f} | {p95:18.3f}")

    # Boş panelleri gizle
    for j in range(n_models, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Alt satıra X ekseni etiketi
    for ax in axes_flat[max(0, n_models - n_cols):n_models]:
        ax.set_xlabel('Time (Hours)')

    print("  " + "-" * 65)

    plt.tight_layout()

    # PNG olarak kaydet
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"satellite_G{sat_prn:02d}_model_errors.png")
    fig.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  [SAVED] Graph saved: {png_path}")

    plt.show()
    plt.close(fig)


def visualize_orbit_comparison(epoch_times: np.ndarray,
                               true_xyz: np.ndarray,
                               results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
                               sat_prn: int,
                               output_dir: str = "output") -> None:
    """
    Her modeli ayrı panelde gösterir (2x3 grid).
    Her panelde orijinal yörünge (kesikli) ile modelin tahmini (düz çizgi) karşılaştırılır.
    X, Y, Z eksenleri ayrı renklerle gösterilir.
    """
    AXIS_COLORS = {'X': '#e74c3c', 'Y': '#2ecc71', 'Z': '#3498db'}
    AXIS_LABELS = ['X', 'Y', 'Z']

    models = list(results_dict.keys())
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), sharex=True)
    fig.suptitle(f'Satellite G{sat_prn:02d} - Orbit Comparison (Original vs Predicted) on SHA',
                 fontsize=15, fontweight='bold')

    axes_flat = np.array(axes).flatten() if n_models > 1 else [axes]

    # Orijinal veriyi saat cinsine çevir
    true_hours = epoch_times / 3600

    for i, name in enumerate(models):
        ax = axes_flat[i]
        p_times, p_xyz = results_dict[name]

        true_interp = _interpolate_truth(p_times, epoch_times, true_xyz)
        p_hours = p_times / 3600
        step = max(1, len(p_times) // 1500)

        for j, axis_name in enumerate(AXIS_LABELS):
            col = AXIS_COLORS[axis_name]

            # Orijinal veri (kesikli çizgi)
            ax.plot(true_hours, true_xyz[:, j], color=col, linestyle='--',
                    linewidth=1.2, alpha=0.6)

            # Model tahmini (düz çizgi)
            ax.plot(p_hours[::step], p_xyz[::step, j], color=col,
                    linewidth=1.5, alpha=0.9, label=f'{axis_name}')

        # RMSE hesapla (3D, metre cinsinden)
        error_3d_m = np.sqrt(np.sum((p_xyz - true_interp) ** 2, axis=1)) * 1000
        rmse = np.sqrt(np.mean(error_3d_m ** 2))

        model_col = MODEL_COLORS.get(name, 'gray')
        ax.set_title(f'{name}  (3D RMSE: {rmse:.0f} m)', fontsize=11,
                     fontweight='bold', color=model_col)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Position (km)')

        # İlk panelde legend göster
        if i == 0:
            # Özel legend: kesikli = orijinal, düz = tahmin
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='gray', linestyle='--', linewidth=1.2, label='Original'),
                Line2D([0], [0], color='gray', linestyle='-', linewidth=1.5, label='Predicted'),
                Line2D([0], [0], color=AXIS_COLORS['X'], linewidth=2, label='X'),
                Line2D([0], [0], color=AXIS_COLORS['Y'], linewidth=2, label='Y'),
                Line2D([0], [0], color=AXIS_COLORS['Z'], linewidth=2, label='Z'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
                      framealpha=0.9, ncol=2)

    # Boş panelleri gizle
    for j in range(n_models, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Alt satıra X ekseni etiketi
    for ax in axes_flat[max(0, n_models - n_cols):n_models]:
        ax.set_xlabel('Time (Hours)')

    plt.tight_layout()

    # PNG olarak kaydet
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"satellite_G{sat_prn:02d}_orbit_comparison.png")
    fig.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  [SAVED] Orbit comparison graph saved: {png_path}")

    plt.show()
    plt.close(fig)
