import numpy as np
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

x_lab = [y for y in data.keys() if 'TP=0' in y ]
x_lab_p = [y[1:-1].split(',')[1].split('=')[1] for y in x_lab]
main_bars = [float(data[x][1]-data[x][0])/10**9 for x in x_lab]
sub_bars = [float(data[x][0])/10**9 for x in x_lab]

colors = plt.cm.Blues(np.linspace(0.2, 1, 3))
plt.bar(x_lab_p, main_bars, width=0.35, label='Other Kernels', color=colors[0])
plt.bar(x_lab_p, sub_bars, width=0.35, label='Communication Kernels', bottom=main_bars)



plt.ylabel('Total Kernel Running Time')

plt.xlabel('Pipeline Parallelism GPU Count')

plt.title('Communication Overhead in Pipeline Parallelism')
plt.legend()

ax2 = plt.twinx()
ax2.plot(x_lab_p, [float(data[y][0])/10**9/int(y[1:-1].split(',')[1].split('=')[1]) for y in x_lab], color='black', marker='o', label='Per GPU Communication Load')
ax2.set_ylim(0, 100)
ax2.set_ylabel('Per GPU Communication Kernel Running Time (sec)')
plt.legend(loc='upper right')
plt.savefig('pp_comm_load.png', dpi=300)

