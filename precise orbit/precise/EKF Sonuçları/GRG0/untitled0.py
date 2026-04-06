#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:39:44 2026

@author: fkarlitepe
"""

input_file = "/home/fkarlitepe/Documents/Projects/Estimation-Orbit/precise orbit/precise/EKF Sonuçları/GRG0/deneme11.sp3"
output_file = "/home/fkarlitepe/Documents/Projects/Estimation-Orbit/precise orbit/precise/EKF Sonuçları/GRG0/deneme_clockfixed.sp3"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []

for line in lines:
    if line.startswith("P") and len(line) > 4:
        sat = line[:4]
        rest = line[4:].split()
        if len(rest) >= 3:
            x = float(rest[0])
            y = float(rest[1])
            z = float(rest[2])
            new_line = f"{sat}{x:14.6f}{y:14.6f}{z:14.6f}{999999.999999:14.6f}\n"
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Oluşturuldu:", output_file)