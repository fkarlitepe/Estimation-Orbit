# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Dict


def parse_sp3(filepath: str, constellation: str = "G") -> Tuple[np.ndarray, Dict[int, np.ndarray], float]:
    """
    SP3 hassas yörünge dosyasını okur ve uydu verilerini ayrıştırır.
    """
    prefix = f"P{constellation}"
    satellite_data: Dict[int, list] = {}
    epoch_count = 0
    header_interval = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("EOF"):
                break

            if line.startswith('*'):
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
                parts = line.split() # Regex yerine standart split (boşluklara göre otomatik ayırır)
                try:
                    prn = int(parts[0][2:])
                    x, y, z, clk = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    if prn not in satellite_data:
                        satellite_data[prn] = []
                    satellite_data[prn].append([x, y, z, clk])
                except (IndexError, ValueError):
                    continue

    epoch_times = np.arange(epoch_count, dtype=float) * header_interval
    final_sat_data = {prn: np.array(data) for prn, data in satellite_data.items()}

    print(f"  [VERİ] SP3 Taraması: {epoch_count} kayıt, {len(final_sat_data)} uydu, "
          f"Aralık={header_interval}s, Toplam={epoch_times[-1]/3600:.1f} saat.")
    
    return epoch_times, final_sat_data, header_interval
