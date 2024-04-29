import numpy as np
import csv
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 10))

with open('../data/latency_throughput.txt') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    rows = []
    for row in csv_reader:
        rows.append({x:y for x, y in zip(header, row)})

data_x, data_y = [], []
labels = []
for x in rows:
    data_x.append(float(x['Throughput']))
    data_y.append(float(x['TotalLatency']))
    labels.append(int(x['BS']))

plt.plot(data_x, data_y, marker='o', color='black')
plt.ylim(0, 3.2)

# Add labels for each point in the line plot
for i, value in enumerate(data_y):
    plt.text(data_x[i], value+0.05, labels[i], ha='center', va='bottom', color='red')  # Add label to each point

plt.xlabel('Throughput (tokens/sec)')
plt.ylabel('Latency')
plt.title('Latency vs Throughput with Batch Size')

plt.savefig('latency_throughput.png', dpi=300)
