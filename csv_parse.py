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
        print(f'{row[0]} & {float(row[1]):.4f} & {float(row[2]):.4f} & {(float(row[1]) - float(data[1][1])):.4f} & {(float(row[2]) - float(data[1][2])):.4f} \\\\ \hline')
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
    psnrs_bar = plt.bar(models, psnrs)
    plt.bar_label(psnrs_bar, fmt='{:.2f}')
    if len(sys.argv) == 4 and sys.argv[3] == 'save':
        plt.savefig(f'./test/{sys.argv[1]}/psnr.png')
    plt.figure(1)
    plt.title('SSIM')
    ssims_bar = plt.bar(models, ssims)
    plt.bar_label(ssims_bar, fmt='{:.2f}')
    if len(sys.argv) == 4 and sys.argv[3] == 'save':
        plt.savefig(f'./test/{sys.argv[1]}/ssim.png')
    plt.figure(2)
    plt.title('MSE')
    mses_bar = plt.bar(models, mses)
    plt.bar_label(mses_bar, fmt='{:.2f}')
    if len(sys.argv) == 4 and sys.argv[3] == 'save':
        plt.savefig(f'./test/{sys.argv[1]}/mse.png')
    if len(sys.argv) == 3:
        plt.show()