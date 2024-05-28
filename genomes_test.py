import datetime
import json
import pickle as pkl
import time
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from pytorch_pre_bert import BertTokenizer
from models.vgcn_bert import VGCNBertModel
from utils import *


def calculate_auroc(TP, TN, FP, FN, scores):
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    sorted_indices = sorted(range(len(TPR)), key=lambda i: FPR[i])
    sorted_TPR = [TPR[i] for i in sorted_indices]
    sorted_FPR = [FPR[i] for i in sorted_indices]

    auroc = 0
    for i in range(1, len(sorted_FPR)):
        auroc += (sorted_FPR[i] - sorted_FPR[i - 1]) * (sorted_TPR[i] + sorted_TPR[i - 1]) / 2

    return auroc


def count(y_list):
    pos = 0
    for i in range(len(y_list)):
        if y_list[i] == 1:
            pos += 1
    return 1 if pos >= 5 else 0


def avg_per_domain(pred_out, y_lens, threshold):
    pred = [0 for _ in range(y_lens)]
    for i in range(len(pred_out)):
        a = pred_out[i]
        for j in range(window_size):
            pred[i + j] += a

    for i in range(window_size):
        pred[i] /= (i + 1)
    for i in range(window_size, y_lens - window_size):
        pred[i] /= window_size
    for i in range(y_lens - window_size, y_lens):
        pred[i] /= (window_size - 1)

    for i in range(y_lens):
        pred[i] = 1 if pred[i] > threshold else 0

    return pred


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


def get_pytorch_dataloader(
        examples,
        tokenizer,
        batch_size,
        shuffle_choice=0,
        classes_weight=None,
        total_resample_size=-1,
):
    ds = CorpusDataset(
        examples, tokenizer, gcn_vocab_map, 400, gcn_embedding_dim
    )
    if shuffle_choice == 0:
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=ds.pad,
        )


label2idx = {"0": 0, "1": 1}
idx2label = {0: "0", 1: "1"}
gcn_vocab_size = 19450
window_size = 64
acc = 0.0
adj_tf_threshold = 0
adj_npmi_threshold = 0.1
gcn_embedding_dim = 64
num_classes = 2

# v_adj = 'pmi'
v_adj = 'tf'
# v_adj = 'all'

# tfidf_mode = 'only_tf'
tfidf_mode = 'tf-idf'

model_file = 'output/binary_eda/binary.pt'
bert_model_scale = "bert-base-uncased"

names = ["vocab_adj_tf",
         "vocab_adj_pmi",
         "vocab_map"]
objects = []
cuda_yes = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_yes else "cpu")

for i in range(len(names)):
    datafile = "data/preprocessed/binary_no_eda/data.%s" % (names[i])
    with open(datafile, "rb") as f:
        objects.append(pkl.load(f, encoding="latin1"))

(gcn_vocab_adj_tf, gcn_vocab_adj, gcn_vocab_map) = tuple(objects)

gcn_vocab_adj.data *= gcn_vocab_adj.data > adj_npmi_threshold
gcn_vocab_adj.eliminate_zeros()

if v_adj == "pmi":
    gcn_vocab_adj_list = [gcn_vocab_adj]
elif v_adj == "tf":
    gcn_vocab_adj_list = [gcn_vocab_adj_tf]
elif v_adj == "all":
    gcn_vocab_adj_list = [gcn_vocab_adj_tf, gcn_vocab_adj]

norm_gcn_vocab_adj_list = []
for i in range(len(gcn_vocab_adj_list)):
    adj = gcn_vocab_adj_list[i]
    adj = normalize_adj(adj)
    norm_gcn_vocab_adj_list.append(sparse_scipy2torch(adj.tocoo()).to(device))
gcn_adj_list = norm_gcn_vocab_adj_list

del gcn_vocab_adj_tf, gcn_vocab_adj, gcn_vocab_adj_list


checkpoint = torch.load(model_file)

model = VGCNBertModel.from_pretrained(
    bert_model_scale,
    state_dict=checkpoint["model_state"],
    gcn_adj_dim=gcn_vocab_size,
    gcn_adj_num=len(gcn_adj_list),
    gcn_embedding_dim=gcn_embedding_dim,
    num_labels=len(label2idx),
)
pretrained_dict = checkpoint["model_state"]
net_state_dict = model.state_dict()
pretrained_dict_selected = {
    k: v for k, v in pretrained_dict.items() if k in net_state_dict
}
net_state_dict.update(pretrained_dict_selected)
model.load_state_dict(net_state_dict)
model.to(device)

print(f"Loaded the pretrain model")

with open('data/PF_to_Domain.json', 'r') as file:
    pf2domain = json.load(file)

with open('data/final_domain_vocab.json') as f:
    tokens = json.load(f)
domains_trans = tokens['domains']

tokenizer = BertTokenizer.from_pretrained(
    bert_model_scale, do_lower_case=True
)

all_bgc_tokens = []
tpr_list = []
tnr_list = []
precision_list = []

log_file = open(f'output/eda_binary/925_6_log.txt', 'w')
y_file = open(f'output/eda_binary/925_6_pred_array.txt', 'w')
log_file.write(f'{datetime.datetime.now()}\nmodel:{model_file},\n'
               f'tf-idf mode:{tfidf_mode}, v_adj={v_adj}, npmi_threshold:{adj_npmi_threshold}\n')

dump_dir = 'data/preprocessed/eda_binary'

for i in range(1, 7):
    domains_file = open(f'data/6_genomes/g{i}.txt', 'r')
    labels_file = open(f'data/6_genomes/l{i}.txt', 'r')

    domains = domains_file.read()
    labels = labels_file.read()

    domains_file.close()
    labels_file.close()

    domains_list = domains.split(';')
    labels_list = labels.split(';')

    domains_array = []
    for d in domains_list:
        if d in pf2domain:
            domains_array.append(domains_trans[pf2domain[d]] - 58)
        else:
            domains_array.append(domains_trans['UNK'] - 58)
    all_bgc_tokens.append(domains_array)

    labels_array = []
    pos = 0
    for l in labels_list:
        labels_array.append(int(l))
        pos += int(l)

    print(f'genome_{i} length:', len(labels_array))

    windows = []
    y = []
    win_pos = 0
    if len(domains_array) <= window_size:
        windows.append(domains_array)
        y.append(count(labels_array))
    else:
        for j in range(len(domains_array) - window_size + 1):
            windows.append(domains_array[j: j + window_size])
            y.append(count(labels_array[j: j + window_size]))
            if y[-1] == 1:
                win_pos += 1
    print(f"pos:{pos} / {len(labels_array)}")
    y_prob = np.eye(len(y), len(label2idx))[y]

    examples = []
    for j, ts in enumerate(windows):
        ex = InputExample(j, ts, confidence=y_prob[i], label=y[i])
        examples.append(ex)

    td = get_pytorch_dataloader(
        examples, tokenizer, 16, shuffle_choice=0
    )

    # pred
    model.eval()
    predict_out = []
    scores = []
    ev_loss = 0
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for c, batch in enumerate(td):
            batch = tuple(t.to(device) for t in batch)
            (
                input_ids,
                input_mask,
                segment_ids,
                _,
                label_ids,
                gcn_swop_eye,
            ) = batch
            score_out, eb = model(
                gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask
            )
            score_out = F.softmax(score_out, dim=-1)
            so_cpu = score_out.cpu()
            for j in range(len(so_cpu)):
                predict_out.append(int(np.argmax(so_cpu[j])))
                scores.append(so_cpu[j][1].item())
            if (c * 16) % (1280 * 3) == 0:
                print(f"{((c * 16) / len(y)):.2f}")

    fixed_pred = predict_out
    y_num = count_continuous_segments(y)
    pred_num = count_continuous_segments(fixed_pred)

    y_log = f'gemones{i}\n'
    y_log += str(fixed_pred)
    y_log += '\n'
    y_file.write(y_log)

    tn = fn = fp = tp = 0
    for k in range(len(y)):
        if fixed_pred[k] == y[k] == 0:
            tn += 1
        elif fixed_pred[k] == y[k] == 1:
            tp += 1
        elif fixed_pred[k] == 1 and y[k] == 0:
            fp += 1
        else:
            fn += 1


    tpr = (tp * 1.0 / (tp + fn))
    tnr = (tn * 1.0 / (tn + fp))
    if (tp + fp) > 0:
        precision = tp * 1.0 / (tp + fp)
    tpr_list.append(tpr)
    tnr_list.append(tnr)
    precision_list.append(precision)
    log = f"genomes_{i}: TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn}\n"
    log += f"tpr:{tpr:.3f}, tnr:{tnr:.3f}\n"
    log += f"precision: {precision:.3f}\n"
    log_file.write(log)
    print(log)
    print(f'{i} Finished....')

log = f"\nAvg tpr: {np.mean(tpr_list):.3f}"
log += f"\nAvg tnr: {np.mean(tnr_list):.3f}"
log += f"\nAvg precision : {np.mean(precision_list):.3f}"

print(log)
log_file.write(log)
log_file.close()
y_file.close()
