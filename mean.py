import pandas as pd

# 读取Excel文件
for metric in ['SD','MI','VIF','AG','CC','SCD','EN','Qabf','SF']:
    excel_file = '..\Metric\Metric_M3FD.xlsx'
    df = pd.read_excel(excel_file, sheet_name=f'{metric}', skiprows=1)

    # 删除包含空值的行和列
    df = df.dropna()


    # 计算每列的平均值
    column_means = df.mean(axis=0)


    print(f"{metric}每列的平均值:")
    with open('output.txt', 'a') as f:
        column_means.to_csv('output.txt',header=False, index=False, sep='\t',mode='a')
        f.write('\n')  # 写入一个回车
    print(column_means)