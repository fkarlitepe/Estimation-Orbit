input_file = "/home/fkarlitepe/Documents/Projects/Estimation-Orbit/precise orbit/precise/EKF Sonuçları/SHA0/deneme1.sp3"
output_file = "/home/fkarlitepe/Documents/Projects/Estimation-Orbit/precise orbit/precise/EKF Sonuçları/SHA0/deneme1_GR_only_fixed.sp3"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Veri kısmını bul
data_start = None
for i, line in enumerate(lines):
    if line.startswith("*"):
        data_start = i
        break

if data_start is None:
    raise ValueError("Epoch satırı (*) bulunamadı.")

header = lines[:data_start]
data = lines[data_start:]

# İlk epoch'tan gerçek uydu listesini çıkar
satellites = []
for line in data:
    if line.startswith("*") and satellites:
        break
    if line.startswith("PG") or line.startswith("PR"):
        satellites.append(line[1:4])

sat_count = len(satellites)

# İlk iki header satırını al
line1 = header[0].rstrip("\n")
line2 = header[1].rstrip("\n")

# 1. satırdaki epoch sayısını doğru yaz
# örnek: #dP2020  2  2  0  0  0.00000000   86400 d+D   IGS14 FIT AIUB
parts = line1.split()
year = parts[0][3:7]
month = int(parts[1])
day = int(parts[2])
hour = int(parts[3])
minute = int(parts[4])
sec = float(parts[5])

new_line1 = f"#dP{year}{month:3d}{day:3d}{hour:3d}{minute:3d}{sec:12.8f}{86400:8d} d+D   IGS14 FIT AIUB"
new_line2 = f"## 2091      0.00000000     1.00000000 58881 0.0000000000000"

# + satırları (17 uydu/satır)
plus_lines = []
chunks = [satellites[i:i+17] for i in range(0, len(satellites), 17)]

for idx, chunk in enumerate(chunks):
    sat_text = "".join(chunk)
    if idx == 0:
        plus_lines.append(f"+{sat_count:5d}   {sat_text:<51}\n")
    else:
        plus_lines.append(f"+        {sat_text:<51}\n")

# Toplam 5 adet + satırı olacak şekilde doldur
while len(plus_lines) < 5:
    plus_lines.append("+        " + "  0"*17 + "\n")

# ++ satırları
pp_lines = []
for i in range(len(plus_lines)):
    if i < (sat_count + 16) // 17:
        n_this = min(17, sat_count - i*17)
        acc = "  5" * n_this + "  0" * (17 - n_this)
    else:
        acc = "  0" * 17
    pp_lines.append("++       " + acc + "\n")

# % ve /* satırlarını koru
tail_header = []
for line in header[2:]:
    if line.startswith("%") or line.startswith("/*"):
        tail_header.append(line)

# Veri kısmında sadece PG/PR/epoch/EOF bırak
filtered_data = []
for line in data:
    if line.startswith("*") or line.startswith("EOF"):
        filtered_data.append(line)
    elif line.startswith("PG") or line.startswith("PR"):
        filtered_data.append(line)

new_lines = []
new_lines.append(new_line1 + "\n")
new_lines.append(new_line2 + "\n")
new_lines.extend(plus_lines)
new_lines.extend(pp_lines)
new_lines.extend(tail_header)
new_lines.extend(filtered_data)

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Yeni dosya oluşturuldu:")
print(output_file)
print("Uydu sayısı:", sat_count)
print("Uydular:", satellites)