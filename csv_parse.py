import csv
import sys
import matplotlib.pyplot as plt

data = []
with open(f'test/{sys.argv[1]}/results.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)

# print(data)
if sys.argv[2] == 'table':
    print('Model & PSNR & SSIM & Poprawa PSNR & Poprawa SSIM \\\\ \hline')
    for row in data:
        if row[0] == 'Model':
            continue
        print(f'{row[0]} & {float(row[1]):.4f} & {float(row[2]):.4f} & {float(data[1][1]) - float(row[1]):.4f} & {float(data[1][2]) - float(row[2]):.4f} \\\\ \hline')
elif sys.argv[2] == 'chart':
    models = []
    psnrs = []
    ssims = []
    mses = []
    for row in data:
        if row[0] == 'Model':
            continue
        models.append(row[0])
        psnrs.append(float(row[1]))
        ssims.append(float(row[2]))
        mses.append(float(row[3]))
    plt.figure(0)
    plt.title('PSNR')
    plt.bar(models, psnrs)
    plt.figure(1)
    plt.title('SSIM')
    plt.bar(models, ssims)
    plt.figure(2)
    plt.title('MSE')
    plt.bar(models, mses)
    plt.show()