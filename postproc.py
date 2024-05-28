import numpy as np
from matplotlib import pyplot as plt


def find_continuous_ones_indices(nums):
    start = None
    end = None
    ones_indices = []

    for i, num in enumerate(nums):
        if num == 1:
            if start is None:
                start = i
            end = i
        else:
            if start is not None:
                ones_indices.append((start, end))
                start = None
                end = None

    if start is not None:
        ones_indices.append((start, end))

    return ones_indices


def win_to_domain(y_win, lens, threshold):
    y = [0 for _ in range(lens)]
    for i in range(len(y_win)):
        if y_win[i] == 1:
            for j in range(i, i + 64):
                y[j] += 1

    for i in range(64):
        y[i] /= (i + 1)
    for i in range(64, lens - 64):
        y[i] /= 64
    for i in range(lens - 64, lens):
        y[i] /= (lens - i)

    score_y = y

    for i in range(lens):
        y[i] = 1 if y[i] >= threshold else 0

    return y, score_y


def count_continuous_segments(arr):
    count = 0
    segment_count = 0
    is_in_segment = False

    for num in arr:
        if num == 1 and not is_in_segment:
            is_in_segment = True
            segment_count += 1
        elif num == 0 and is_in_segment:
            is_in_segment = False

        if is_in_segment:
            count += 1

    return segment_count


def merge_ones(arr, k, v):
    merge = [0 for _ in range(len(arr))]
    last_one_index = -1
    for i in range(len(arr)):
        l = arr[i]
        if l == 0:
            merge[i] = 0
        elif l == 1 and last_one_index != -1:
            merge[i] = 1
            if i - last_one_index <= k:
                for j in range(last_one_index, i):
                    merge[j] = 1
            last_one_index = i
        else:
            merge[i] = 1
            last_one_index = i
    delect = merge
    count = 0
    for i in range(len(delect)):
        if delect[i] == 1:
            count += 1
        else:
            if count > 0 and count < v:
                for j in range(i - count, i):
                    delect[j] = 0
            count = 0

    if count > 0 and count < v:
        for j in range(len(delect) - count, len(delect)):
            delect[j] = 0

    return delect


def compute_metric(p, l):
    tp_ = tn_ = fp_ = fn_ = 0
    for j in range(len(p)):
        if l[j] == 1 and l[j] == p[j]:
            tp_ += 1
        elif l[j] == 0 and y[j] == p[j]:
            tn_ += 1
        elif l[j] == 1:
            fn_ += 1
        else:
            fp_ += 1

    pr = 0 if tp_ + fp_ == 0 else tp_ / (tp_ + fp_)
    re = 0 if tp_ + fn_ == 0 else tp_ / (tp_ + fn_)
    omi = 0 if tp_ + tn_ == 0 else fp_ / (fp_ + tn_)
    return pr, re, omi


domain_tpr_list = []
domain_tnr_list = []
domain_acc_list = []
domain_prec_list = []
domain_fpr_list = []
domain_fnr_list = []
genome_length = []

true_labels = []

min_pl = 1000000
colors = ['#EAD1DC', '#FFE156', '#B0C4DE', '#DD9977', '#55AAAA', '#CC6677', '#882255']
start_end = []

ours1 = ours5 = ours8 = deep = cf = [0, 0, 0]

# pf -> num
for i in range(1, 7):
    se = []
    y_file = open(f'../../data/6_genomes/l{i}.txt')
    y_win_file = open(f'../../data/6_genomes/g{i}_window_true_labels.txt', 'r')
    pred_win_file = open(f'../../data/6_genomes/111eda{i}.txt', 'r')

    y = y_file.read()
    y_domain = y.split(';')
    y_domain_len = len(y_domain)
    y_win = eval(y_win_file.read())
    pred = eval(pred_win_file.read())

    pred_domain, yscore = win_to_domain(pred, y_domain_len, 1)
    pred_domain5, yscore5 = win_to_domain(pred, y_domain_len, 0.5)
    pred_domain8, yscore8 = win_to_domain(pred, y_domain_len, 0.8)
    fixed_pred_domain = merge_ones(pred_domain, 10, 5)
    fixed_pred_domain5 = merge_ones(pred_domain5, 10, 5)
    fixed_pred_domain8 = merge_ones(pred_domain8, 10, 5)

    y_array = []
    for l in y_domain:
        y_array.append(int(l))

    y_domain_num = count_continuous_segments(y_array)
    pred_domain_num = count_continuous_segments(fixed_pred_domain)

    print('--------------------------Domains--------------------------')
    print(
        f'genome_{i} length: {len(fixed_pred_domain)}, True_BGC_nums: {y_domain_num}, Pred_BGC_nums: {pred_domain_num}')
    tp = tn = fp = fn = 0
    t = 0
    genome_length.append(len(fixed_pred_domain))
    for j in range(len(y_domain)):
        if y_array[j] == 1 and y_array[j] == fixed_pred_domain[j]:
            tp += 1
            t += 1
        elif y_array[j] == 0 and y_array[j] == fixed_pred_domain[j]:
            tn += 1
            t += 1
        elif y_array[j] == 1:
            fn += 1
        else:
            fp += 1

    tpr = 0 if tp + fn == 0 else tp / (tp + fn)
    tnr = 0 if tn + fp == 0 else tn / (tn + fp)
    fpr = 0 if fp + tn == 0 else fp / (fp + tn)
    fnr = 0 if tp + fn == 0 else fn / (tp + fn)
    acc = t / len(y_array)
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    domain_tpr_list.append(tpr)
    domain_tnr_list.append(tnr)
    domain_acc_list.append(acc)
    domain_fpr_list.append(fpr)
    domain_fnr_list.append(fnr)
    domain_prec_list.append(precision)

    print(
        f'Genome{i} fnr: {fnr:.3f}, fpr: {fpr:.3f}, prec:{precision:.3f}, acc: {acc:.3f}')

    pp = find_continuous_ones_indices(fixed_pred_domain)
    se.append(pp)
    t_l = find_continuous_ones_indices(y_array)
    true_labels.append(t_l)

    pp = find_continuous_ones_indices(fixed_pred_domain8)
    se.append(pp)

    pp = find_continuous_ones_indices(fixed_pred_domain5)
    se.append(pp)

    with open(f'../../data/6_genomes/deepbgc_pred{i}.txt') as f:
        op = f.read()
        opp = eval(op)
        dpp = find_continuous_ones_indices(opp)
        se.append(dpp)

    with open(f'../../data/6_genomes/cf_pred{i}.txt') as f:
        op = f.read()
        opp = eval(op)
        dpp = find_continuous_ones_indices(opp)
        se.append(dpp)

    start_end.append(se)

print(
    f'\nDomain_Mean fnr: {np.mean(domain_fnr_list):.3f}, fpr: {np.mean(domain_fpr_list):.3f}, prec:{np.mean(domain_prec_list):.3f}, acc: {np.mean(domain_acc_list):.3f}')


sequence_titles = ['NC_007413.1', 'NC_004129.6', 'AM420293.1', 'AP009493.1', 'NC_009380.1', 'AM746676.1']
detector_labels = ['BGCCGB(thr=1)', 'BGCCGB(thr=0.8)', 'BGCCGB(thr=0.5)', 'DeepBGC', 'ClusterFinder']
num_sequences = len(sequence_titles)
num_detectors = len(detector_labels)

fig, axes = plt.subplots(num_sequences, 1, figsize=(15, 1 + 0.25 * (num_detectors + 2) * num_sequences))
if num_sequences == 1:
    axes = [axes]

for i, (clusters, sequence_title) in enumerate(zip(start_end, sequence_titles)):
    ax = axes[i]
    ax.set_facecolor('white')
    ax.set_yticks(range(1, num_detectors + 1))
    ax.set_yticklabels(reversed(detector_labels))
    ax.set_ylim([0.3, num_detectors + 0.7])
    ax.set_title(sequence_title)

    end = genome_length[i]
    x_step = 1000
    if end / x_step > 20:
        x_step = 5000
    if end / x_step > 20:
        x_step = 10000
    if end / x_step > 20:
        x_step = 20000

    cmap = plt.get_cmap("tab10")

    color = 'grey'
    for cluster in true_labels[i]:
        ax.axvspan(cluster[0], cluster[1], color=color, alpha=0.3)

    color_idx = 0
    for level, detector_label in enumerate(detector_labels):
        color = cmap(color_idx)
        color_idx += 1

        x = [item for sublist in start_end[i][level] for item in sublist]
        y = np.ones([len(x), 2]) * (num_detectors - level)  # 5, 4, 3, 2, 1
        y[1::2] = np.nan
        for d in np.arange(-0.08, 0.08, 0.005):
            ax.step(x, y + d, color=colors[level + 2], where='post', lw=0.6, label=None)

        ax.set_xlabel('')
        xticks = range(0, genome_length[i] + x_step, x_step)
        ax.set_xticks(xticks)

axes[-1].set_xlabel('BGC Candidate')
fig.tight_layout()
fig.savefig(f'figures/ours.png', dpi=500, bbox_inches='tight')
fig.show()
