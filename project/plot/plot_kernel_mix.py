import csv
import matplotlib.pyplot as plt

with open('../data/nccl_kernel_sum/communication_kernel_sum.txt', 'r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    rows = []
    for row in csv_reader:
        rows.append({x:y for x, y in zip(header, row)})
    
data = {}
for row in rows:
    if int(row['BS']) != 32:
        continue
    tm = float(row['CommKernelTime']) * 40 / int(row['Iterations'])
    pct = float(row['CommKernelPct'])
    x = tm * 100.0 / pct
    data['(TP={},PP={})'.format(row['TP'], row['PP'])] = (int(tm), int(x))

x_lab = data.keys()
main_bars = [float(data[x][1]-data[x][0])/10**9 for x in x_lab]
sub_bars = [float(data[x][0])/10**9 for x in data]

plt.bar(x_lab, main_bars, width=0.35, label='Other Kernels')
plt.bar(x_lab, sub_bars, width=0.35, label='Communication Kernels', bottom=main_bars)

plt.ylabel('Total Kernel Running Time')

plt.xlabel('Configuration')

plt.title('Kernel Running Times')
plt.legend()
plt.savefig('kernel_mix_plot.png', dpi=300)

