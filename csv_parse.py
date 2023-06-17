import csv
import sys

data = []
with open(f'test/{sys.argv[1]}/results.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)

# print(data)
print('Model & PSNR & SSIM & Poprawa PSNR & Poprawa SSIM \\\\ \hline')
for row in data:
    if row[0] == 'Model':
        continue
    print(f'{row[0]} & {float(row[1]):.4f} & {float(row[2]):.4f} & {float(data[1][1]) - float(row[1]):.4f} & {float(data[1][2]) - float(row[2]):.4f} \\\\ \hline')