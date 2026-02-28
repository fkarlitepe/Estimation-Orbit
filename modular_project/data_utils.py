# -*- coding: utf-8 -*-
import re
import numpy as np
from typing import Tuple, Dict, List, Optional


def parse_sp3(filepath: str, constellation: str = "G") -> Tuple[np.ndarray, Dict[int, np.ndarray], float]:
    """
    SP3 hassas yörünge dosyasını okur ve uydu verilerini ayrıştırır.
    Gece yarısı geçiş (rollover) problemlerini önlemek için saniye bazlı indeksleme yapar.
    """
    prefix = f"P{constellation}"
    satellite_data: Dict[int, List[List[float]]] = {}
    current_epoch_idx = -1
    epoch_count = 0
    header_interval = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("EOF"):
                break

            if line.startswith('*'):
                current_epoch_idx += 1
                epoch_count += 1
                
                # Başlıktan ölçüm aralığını (interval) saniye cinsinden bulma
                if header_interval is None and epoch_count == 1:
                    parts = line.split()
                    first_h, first_m, first_s = int(parts[4]), int(parts[5]), float(parts[6])
                elif header_interval is None and epoch_count == 2:
                    parts = line.split()
                    second_h, second_m, second_s = int(parts[4]), int(parts[5]), float(parts[6])
                    dt = (second_h - first_h) * 3600 + (second_m - first_m) * 60 + (second_s - first_s)
                    if dt <= 0: dt += 86400  # Gece yarısı geçişi için +24 saat
                    header_interval = dt
                continue

            if line.startswith(prefix):
                clean_line = re.sub(r'\s+', ' ', line).strip()
                parts = clean_line.split(' ')
                try:
                    prn = int(parts[0][2:])
                    x, y, z, clk = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    if prn not in satellite_data:
                        satellite_data[prn] = []
                    satellite_data[prn].append([x, y, z, clk])
                except (IndexError, ValueError):
                    continue

    if header_interval is None or header_interval <= 0:
        header_interval = 300.0

    epoch_times = np.arange(epoch_count, dtype=float) * header_interval
    
    # Listeleri NumPy dizilerine dönüştür
    final_sat_data = {prn: np.array(data) for prn, data in satellite_data.items()}

    print(f"  [VERİ] SP3 Taraması: {epoch_count} kayıt, {len(final_sat_data)} uydu, "
          f"Aralık={header_interval}s, Toplam={epoch_times[-1]/3600:.1f} saat.")
    
    return epoch_times, final_sat_data, header_interval


def write_sp3(filepath: str, 
              predict_times: np.ndarray, 
              satellite_results: Dict[int, Tuple[np.ndarray, np.ndarray]], 
              epoch_interval: float,
              constellation: str = "G", 
              ref_date: Tuple[int, int, int] = (2020, 3, 30)) -> None:
    """
    Tahmin edilen yörünge sonuçlarını standart SP3c formatında veritabanı olarak kaydeder.
    """
    year, month, day = ref_date
    n_sats = len(satellite_results)
    n_epochs = len(predict_times)
    prns = sorted(satellite_results.keys())

    with open(filepath, 'w') as f:
        # Başlık Bilgileri (Header)
        f.write(f"#dP{year}  {month:2d} {day:2d}  0  0  0.00000000    {n_epochs:5d} d+D   IGS14 FIT EST\n")
        f.write(f"## {0:4d}      0.00000000 {epoch_interval:14.8f} {0:5d} 0.0000000000000\n")

        # Uyduların Tanımlanması (Max 17 uydu per line)
        sat_ids = [f"{constellation}{p:02d}" for p in prns]
        for line_idx in range(5):
            start = line_idx * 17
            line_sats = sat_ids[start:min(start + 17, len(sat_ids))]
            prefix = "+" if line_idx == 0 else "+"
            sat_str = "".join(line_sats).ljust(17*3)
            f.write(f"{prefix}   {n_sats if line_idx == 0 else '  '}   {sat_str}\n")

        # Opsiyonel Tanımlama Satırları (SP3 format gereği sabit doldurulur)
        for _ in range(2): f.write("++         " + "  5" * 17 + "\n")
        f.write("%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n")
        f.write("%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n")
        f.write("/* Modular Project Estimated Orbits\n")

        # Her Zaman ve Her Uydu İçin Kayıt Yazma
        for epoch_idx, t_sec in enumerate(predict_times):
            h, m, s = int(t_sec // 3600), int((t_sec % 3600) // 60), t_sec % 60
            h_wrap, d_plus = h % 24, h // 24
            f.write(f"*  {year}  {month:2d} {day + d_plus:2d} {h_wrap:2d} {m:2d} {s:11.8f}\n")
            
            for prn in prns:
                xyz_km, clk_us = satellite_results[prn]
                x, y, z = xyz_km[epoch_idx]
                c = clk_us[epoch_idx, 0]
                f.write(f"P{constellation}{prn:02d}{x:14.6f}{y:14.6f}{z:14.6f}{c:14.6f}\n")
        
        f.write("EOF\n")
    print(f"  [KAYIT] SP3 dosyası oluşturuldu: {filepath}")
