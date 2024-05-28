import argparse
import gc
import json
import os
import pickle as pkl
import random
import time
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pre_bert.optimization import BertAdam
from pytorch_pre_bert.tokenization import BertTokenizer
from sklearn.metrics import classification_report, f1_score
from env_config import env_config
from models.vgcn_bert import VGCNBertModel
from utils import *

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

cuda_yes = torch.cuda.is_available()
if cuda_yes:
    torch.cuda.manual_seed_all(42)
device = torch.device("cuda:0" if cuda_yes else "cpu")

lr = 0.00001
l2 = 0.5
dropout_rate = 0.5
parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, default="bgc")
parser.add_argument("--load", type=int, default=0)
parser.add_argument("--sw", type=int, default="0")
parser.add_argument("--dim", type=int, default="0")
parser.add_argument("--lr", type=float, default=lr)
parser.add_argument("--l2", type=float, default=l2)
parser.add_argument("--model", type=str, default="VGCN_BERT")
parser.add_argument("--validate_program", action="store_true")
args = parser.parse_args()
args.ds = args.ds
cfg_model_type = args.model
gcn_embedding_dim = args.dim
learning_rate0 = args.lr
l2_decay = args.l2

log_file = open(f'output/mutil/1023output.txt', 'w')
will_train_mode_from_checkpoint = False
total_train_epochs = 50

batch_size = 16

MAX_SEQ_LENGTH = 500 + gcn_embedding_dim
gradient_accumulation_steps = 1

bert_model_scale = "bert-base-uncased"
if env_config.TRANSFORMERS_OFFLINE == 1:
    bert_model_scale = os.path.join(
        env_config.HUGGING_LOCAL_MODEL_FILES_PATH,
        f"hf-maintainers_{bert_model_scale}",
    )

do_lower_case = True
warmup_proportion = 0.0

data_dir = f"data/preprocessed/"
output_dir = "output/mutil"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

perform_metrics_str = ["weighted avg", "f1-score"]

# cfg_vocab_adj = "pmi"
# cfg_vocab_adj = 'all'
cfg_vocab_adj = 'tf'
tf_mode = 'only_tf'
# tf_mode = 'tf-idf'
cfg_adj_npmi_threshold = 0.1
cfg_adj_tf_threshold = 0
classifier_act_func = nn.ReLU()

do_softmax_before_mse = True
cfg_loss_criterion = "mse"
model_file_4save = (
    f"mutil_classification.pt"
)

if args.validate_program:
    total_train_epochs = 1

print(cfg_model_type + " Start at:", time.asctime())
print(
    "\n----- Configure -----",
    f"\n  Vocab GCN_hidden_dim: vocab_size -> 128 -> {str(gcn_embedding_dim)}",
    f"\n  Learning_rate0: {learning_rate0}" f"\n  weight_decay: {l2_decay}",
    f"\n  Loss_criterion {cfg_loss_criterion}"
    f"\n  softmax_before_mse: {do_softmax_before_mse}",
    f"\n  Dropout: {dropout_rate}"
    f"\n  gcn_act_func: Relu",
    f"\n  model_file_4save: {model_file_4save}"
)

objects = []
names = [
    "labels",
    "train_y",
    "train_y_prob",
    "valid_y",
    "valid_y_prob",
    "test_y",
    "test_y_prob",
    "all_BGC_tokens",
    "vocab_adj_tf",
    "vocab_adj_pmi",
    "vocab_map",
]
for i in range(len(names)):
    datafile = data_dir + "mutil/data.%s" % (names[i])
    with open(datafile, "rb") as f:
        objects.append(pkl.load(f, encoding="latin1"))
(
    lables_list,
    train_y,
    train_y_prob,
    valid_y,
    valid_y_prob,
    test_y,
    test_y_prob,
    all_BGC_tokens,
    gcn_vocab_adj_tf,
    gcn_vocab_adj,
    gcn_vocab_map,
) = tuple(objects)

train_y = np.array(train_y)
valid_y = np.array(valid_y)
test_y = np.array(test_y)

label2idx = lables_list[0]
idx2label = lables_list[1]

y = np.hstack((train_y, valid_y, test_y))
y_prob = np.vstack((train_y_prob, valid_y_prob, test_y_prob))

examples = []
for i, ts in enumerate(all_BGC_tokens):
    ex = InputExample(i, ts, confidence=y_prob[i], label=y[i])
    examples.append(ex)

num_classes = len(label2idx)
gcn_vocab_size = 19450
train_size = len(train_y)
valid_size = len(valid_y)
test_size = len(test_y)


indexs = np.arange(0, len(examples))
train_examples = [examples[i] for i in indexs[:train_size]]
valid_examples = [
    examples[i] for i in indexs[train_size: train_size + valid_size]
]
test_examples = [
    examples[i]
    for i in indexs[
             train_size + valid_size: train_size + valid_size + test_size
             ]
]

if cfg_adj_tf_threshold > 0:
    gcn_vocab_adj_tf.data *= gcn_vocab_adj_tf.data > cfg_adj_tf_threshold
    gcn_vocab_adj_tf.eliminate_zeros()
if cfg_adj_npmi_threshold > 0:
    gcn_vocab_adj.data *= gcn_vocab_adj.data > cfg_adj_npmi_threshold
    gcn_vocab_adj.eliminate_zeros()

if cfg_vocab_adj == "pmi":
    gcn_vocab_adj_list = [gcn_vocab_adj]
elif cfg_vocab_adj == "tf":
    gcn_vocab_adj_list = [gcn_vocab_adj_tf]
elif cfg_vocab_adj == "all":
    gcn_vocab_adj_list = [gcn_vocab_adj_tf, gcn_vocab_adj]

norm_gcn_vocab_adj_list = []
for i in range(len(gcn_vocab_adj_list)):
    adj = gcn_vocab_adj_list[i]
    adj = normalize_adj(adj)
    norm_gcn_vocab_adj_list.append(sparse_scipy2torch(adj.tocoo()).to(device))
gcn_adj_list = norm_gcn_vocab_adj_list

del gcn_vocab_adj_tf, gcn_vocab_adj, gcn_vocab_adj_list
gc.collect()

train_classes_num, train_classes_weight = get_class_count_and_weight(
    train_y, len(label2idx)
)
loss_weight = torch.tensor(train_classes_weight, dtype=torch.float).to(device)

tokenizer = BertTokenizer.from_pretrained(
    bert_model_scale, do_lower_case=do_lower_case
)


def count_RnP(pred, labels):
    counts = correct = predicted = [0, 0, 0, 0, 0, 0, 0]
    recall = precision = [0, 0, 0, 0, 0, 0, 0]
    for true_label, predicted_label in zip(labels, pred):
        counts[true_label] += 1  # 真实类别计数加1
        predicted[predicted_label] += 1  # 预测类别计数加1
        if true_label == predicted_label:
            correct[true_label] += 1  # 预测正确的类别计数加1

    # 计算每个类别的 Precision 和 Recall
    for class_label in range(7):
        precision[class_label] = (correct[class_label] / predicted[class_label] if predicted[class_label] != 0 else 0)
        recall[class_label] = (correct[class_label] / counts[class_label] if counts[class_label] != 0 else 0)

    return recall, precision


def get_pytorch_dataloader(
        examples,
        tokenizer,
        batch_size,
        shuffle_choice,
        classes_weight=None,
        total_resample_size=-1,
):
    ds = CorpusDataset(
        examples, tokenizer, gcn_vocab_map, MAX_SEQ_LENGTH, gcn_embedding_dim
    )
    if shuffle_choice == 0:
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=ds.pad,
        )
    elif shuffle_choice == 1:
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=ds.pad,
        )


if args.validate_program:
    train_examples = [train_examples[0]]
    valid_examples = [valid_examples[0]]
    test_examples = [test_examples[0]]

train_dataloader = get_pytorch_dataloader(
    train_examples, tokenizer, batch_size, shuffle_choice=1
)
valid_dataloader = get_pytorch_dataloader(
    valid_examples, tokenizer, batch_size, shuffle_choice=0
)
test_dataloader = get_pytorch_dataloader(
    test_examples, tokenizer, batch_size, shuffle_choice=0
)

total_train_steps = int(
    len(train_dataloader) / gradient_accumulation_steps * total_train_epochs
)


def evaluate(
        model, gcn_adj_list, predict_dataloader, batch_size, epoch_th, dataset_name
):
    model.eval()
    all_label_ids = []
    predict_out = []
    ev_loss = 0
    start = time.time()
    with torch.no_grad():
        count = 0
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            (
                input_ids,
                input_mask,
                segment_ids,
                y_prob,
                label_ids,
                gcn_swop_eye,
            ) = batch
            logits, emb = model(
                gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask
            )

            if cfg_loss_criterion == "mse":
                if do_softmax_before_mse:
                    logits = F.softmax(logits, -1)
                loss = F.mse_loss(logits, y_prob)

            else:
                if loss_weight is None:
                    loss = F.cross_entropy(
                        logits.view(-1, num_classes), label_ids
                    )
                else:
                    loss = F.cross_entropy(
                        logits.view(-1, num_classes), label_ids
                    )
            ev_loss += loss.item()

            _, predicted = torch.max(logits, -1)

        f1_metrics = f1_score(
            np.array(all_label_ids).reshape(-1),
            np.array(predict_out).reshape(-1),
            average="weighted",
        )

    end = time.time()
    return ev_loss, f1_metrics


print("\n----- Running training -----")
if will_train_mode_from_checkpoint and os.path.exists(
        'output/mutil_eda/model_only_tf_npmi0.3_tf0_adj_tf.pt'
):
    checkpoint = torch.load(
        'output/mutil_eda/model_only_tf_npmi0.3_tf0_adj_tf.pt', map_location="cpu"
    )
    if "step" in checkpoint:
        prev_save_step = checkpoint["step"]
        start_epoch = checkpoint["epoch"]
    else:
        prev_save_step = -1
        start_epoch = checkpoint["epoch"] + 1
    valid_acc_prev = checkpoint["valid_acc"]
    perform_metrics_prev = checkpoint["perform_metrics"]
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
    print(
        f"Loaded the pretrain model: {model_file_4save}",
        f", epoch: {checkpoint['epoch']}",
        f"step: {prev_save_step}",
        f"valid acc: {checkpoint['valid_acc']}",
        f"{' '.join(perform_metrics_str)}_valid: {checkpoint['perform_metrics']}",
    )

else:
    start_epoch = 0
    valid_acc_prev = 0
    perform_metrics_prev = 0
    model = VGCNBertModel.from_pretrained(
        bert_model_scale,
        gcn_adj_dim=gcn_vocab_size,
        gcn_adj_num=len(gcn_adj_list),
        gcn_embedding_dim=gcn_embedding_dim,
        num_labels=len(label2idx),
    )
    prev_save_step = -1

model.to(device)

optimizer = BertAdam(
    model.parameters(),
    lr=lr,
    warmup=warmup_proportion,
    t_total=total_train_steps,
    weight_decay=l2,
)

train_start = time.time()
global_step_th = int(
    len(train_examples)
    / batch_size
    / gradient_accumulation_steps
    * start_epoch
)

best_loss = 99999
for epoch in range(start_epoch, total_train_epochs):
    tr_loss = 0
    ep_train_start = time.time()
    model.train()
    optimizer.zero_grad()
    for step, batch in enumerate(train_dataloader):
        if prev_save_step > -1:
            if step <= prev_save_step:
                continue
        if prev_save_step > -1:
            prev_save_step = -1
        batch = tuple(t.to(device) for t in batch)
        (
            input_ids,
            input_mask,
            segment_ids,
            y_prob,
            label_ids,
            gcn_swop_eye,
        ) = batch

        logits, emb = model(
            gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask
        )

        if cfg_loss_criterion == "mse":
            if do_softmax_before_mse:
                logits = F.softmax(logits, -1)
            loss = F.mse_loss(logits, y_prob)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()

        tr_loss += loss.item()
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step_th += 1

    log = "\n--------------------------------------------------------------"

    valid_loss, perform_metrics = evaluate(
        model, gcn_adj_list, valid_dataloader, batch_size, epoch, "Valid_set"
    )

    if best_loss > valid_loss:
        best_loss = valid_loss
        torch.save(model, os.path.join(output_dir, model_file_4save))

    if perform_metrics > perform_metrics_prev:
        to_save = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "lower_case": do_lower_case,
            "perform_metrics": perform_metrics,
        }
        # torch.save(to_save, os.path.join(output_dir, model_file_4save))
        perform_metrics_prev = perform_metrics
        valid_f1_best_epoch = epoch

    print("\nEpoch:{} completed, Total Train Loss:{:.4f}, Valid Loss:{:.4f},Spend {:.2f}m "
          .format(epoch, tr_loss, valid_loss, (time.time() - train_start) / 60.0))


def test_model(
        model, gcn_adj_list, predict_dataloader, batch_size, dataset_name
):
    log = "\n"
    model.eval()
    predict_out = []
    all_label_ids = []
    auroc_list = [0, 0, 0, 0, 0, 0, 0]
    recall_list = [0, 0, 0, 0, 0, 0, 0]
    precision_list = [0, 0, 0, 0, 0, 0, 0]
    ev_loss = 0
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        count = 0
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            (
                input_ids,
                input_mask,
                segment_ids,
                y_prob,
                label_ids,
                gcn_swop_eye,
            ) = batch
            # the parameter label_ids is None, model return the prediction score
            logits, emb = model(
                gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask
            )

            if cfg_loss_criterion == "mse":
                if do_softmax_before_mse:
                    logits = F.softmax(logits, -1)
                loss = F.mse_loss(logits, y_prob)
                l = change(logits)
                y_p = change(y_prob)

                for cls in range(num_classes):
                    y_true_cls = y_prob[:, cls].cpu().numpy()  # 真实标签的某个类别的概率
                    if len(np.unique(y_true_cls)) > 1:  # 检查是否有多种类别
                        y_scores_cls = logits[:, cls].detach().cpu().numpy()  # 模型的预测得分
                        auroc = roc_auc_score(y_true_cls, y_scores_cls)
                        auroc_list[cls] += auroc
                count += 1
            else:
                if loss_weight is None:
                    loss = F.cross_entropy(
                        logits.view(-1, num_classes), label_ids
                    )
                else:
                    loss = F.cross_entropy(
                        logits.view(-1, num_classes), label_ids
                    )
            ev_loss += loss.item()

            _, predicted = torch.max(logits, -1)
            predict_out.extend(predicted.tolist())
            all_label_ids.extend(label_ids.tolist())
            eval_accuracy = predicted.eq(label_ids).sum().item()
            total += len(label_ids)
            correct += eval_accuracy

        f1_metrics = f1_score(
            np.array(all_label_ids).reshape(-1),
            np.array(predict_out).reshape(-1),
            average="weighted",
        )

        log += "\nReport:\n" + classification_report(
            np.array(all_label_ids).reshape(-1),
            np.array(predict_out).reshape(-1),
            digits=4,
            target_names=['Alkaloid', 'NRP', 'Other', 'Polyketide', 'RiPP', 'Saccharide', 'Terpene']
        )

    if cfg_loss_criterion == "mse":
        for i in range(num_classes):
            auroc_list[i] /= count
            recall_list[i] /= count
            precision_list[i] /= count
            
    ev_acc = correct / total
    end = time.time()
    log += f"\n{perform_metrics_str}: {100 * f1_metrics:.2f} Acc : {100.0 * ev_acc:.2f} " \
           f" Spend: {(end - start) / 60.0:.2f} minutes for evaluation"
    log += f'\nAvg AUROC: {np.mean(auroc_list):.3f}'
    log += f"\nAUROC: {auroc_list[0]:.3f}, {auroc_list[1]:.3f}, {auroc_list[2]:.3f}, {auroc_list[3]:.3f}, {auroc_list[4]:.3f}" \
           f" {auroc_list[5]:.3f}, {auroc_list[6]:.3f}"

    print(log)


model = torch.load(os.path.join(output_dir, model_file_4save))
model.to(device)

test_model(model, gcn_adj_list, test_dataloader, batch_size, "Test_set")

log_file.close()
