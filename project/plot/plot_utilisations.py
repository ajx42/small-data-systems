import matplotlib.pyplot as plt
import os

def plot_util(filename, title, outfile):
    plt.figure(figsize=(16, 9))
    ap = []
    pre_ts = 0
    with open(filename) as pp:
        for line in pp:
            line = line.strip()
            if 'node' in line:
                ts_raw = line.strip().split()[4]
                ts_ls = [int(x) for x in ts_raw.split(':')]
                ts = (ts_ls[0]*3600 + ts_ls[1]*60 + ts_ls[2])*1000
                if ts == pre_ts:
                    ts += 300
                pre_ts = ts
            if 'root' in line:
                ap.append((ts, line))

    bp = ap
    exp = 0
    vals = [[] for _ in range(4)]
    ts_x = []
    while bp:
        gpu = int(bp[0][1][1])
        if len(ts_x) == 0 or ts_x[-1] != bp[0][0]:
            ts_x.append(bp[0][0])
        if gpu != exp:
            vals[exp].append(0)
        else:
            vals[exp].append(int(bp[0][1].split(',')[1].split('%')[0].strip()))
            bp = bp[1:]
        exp = (exp + 1) % 4

    if exp != 0:
        for i in range(exp, 4):
            vals[i].append(0)

    plt.title(title)
    plt.stackplot([float(x - min(ts_x))/1000 for x in ts_x] + [(max(ts_x)-min(ts_x))//1000+1], vals[0]+[0], vals[1]+[0], vals[2]+[0], vals[3]+[0], labels=['GPU 1', 'GPU 2', 'GPU 3', 'GPU 4'])
    plt.legend(loc='upper left')
    plt.savefig(outfile, dpi=300)

if __name__ == '__main__':
    plot_util('../data/utilisations/falcon_bs32_pp4.stats.log', 'Utilisations: PP=4', 'plot_bs32_pp4.png')
    plot_util('../data/utilisations/falcon_bs32_tp4.stats.log', 'Utilisations: TP=4', 'plot_bs32_tp4.png')
    plot_util('../data/utilisations/falcon_bs32_tp2_pp2.stats.log', 'Utilisations: TP=2, PP=2', 'plot_bs32_tp2_pp2.png')
