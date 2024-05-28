import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    WeightedRandomSampler,
)


def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def sparse_scipy2torch(coo_sparse):
    row = coo_sparse.row.astype(np.int64)
    col = coo_sparse.col.astype(np.int64)
    i = torch.LongTensor(np.vstack((row, col)))
    v = torch.from_numpy(coo_sparse.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(coo_sparse.shape))


def get_class_count_and_weight(y, n_classes):
    classes_count = []
    weight = []
    for i in range(n_classes):
        count = 0
        for l in y:
            if l == i:
                count += 1
        classes_count.append(count)
        weight.append(len(y) / (n_classes * count))
    return classes_count, weight


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, confidence=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.confidence = confidence
        self.label = label


class InputFeatures(object):
    def __init__(
            self,
            guid,
            tokens,
            input_ids,
            gcn_vocab_ids,
            input_mask,
            segment_ids,
            confidence,
            label_id,
    ):
        self.guid = guid
        self.tokens = tokens
        self.input_ids = input_ids
        self.gcn_vocab_ids = gcn_vocab_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.confidence = confidence
        self.label_id = label_id


def example2feature(
        example, tokenizer, gcn_vocab_map, max_seq_len, gcn_embedding_dim
):
    tokens_a = example.text_a
    assert example.text_b == None
    gcn_vocab_ids = []
    for w in tokens_a:
        gcn_vocab_ids.append(w)
    if type(tokens_a) != list:
        print(tokens_a)
    tokens = (
            ["[CLS]"] + tokens_a + ["[SEP]" for i in range(gcn_embedding_dim + 1)]
    )
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    feat = InputFeatures(
        guid=example.guid,
        tokens=tokens,
        input_ids=input_ids,
        gcn_vocab_ids=gcn_vocab_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        confidence=example.confidence,
        label_id=example.label,
    )
    return feat


class CorpusDataset(Dataset):
    def __init__(
            self,
            examples,
            tokenizer,
            gcn_vocab_map,
            max_seq_len,
            gcn_embedding_dim,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.gcn_embedding_dim = gcn_embedding_dim
        self.gcn_vocab_map = gcn_vocab_map

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat = example2feature(
            self.examples[idx],
            self.tokenizer,
            self.gcn_vocab_map,
            self.max_seq_len,
            self.gcn_embedding_dim,
        )
        return (
            feat.input_ids,
            feat.input_mask,
            feat.segment_ids,
            feat.confidence,
            feat.label_id,
            feat.gcn_vocab_ids,
        )

    def cls_change(self, batch):
        for l in batch:
            s = l[0]
            for i in range(len(s)):
                if np.array_equal(s[i], '[CLS]'):
                    s[i] = 0
                elif np.array_equal(s[i], '[SEP]'):
                    s[i] = 1
        return batch

    def pad(self, batch):
        gcn_vocab_size = 19450
        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = np.array(seqlen_list).max()
        batch = self.cls_change(batch)

        f_collect = lambda x: [sample[x] for sample in batch]
        f_pad = lambda x, seqlen: [
            sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch
        ]
        f_pad2 = lambda x, seqlen: [
            [-1] + sample[x] + [-1] * (seqlen - len(sample[x]) - 1)
            for sample in batch
        ]

        batch_input_ids = torch.tensor(f_pad(0, maxlen), dtype=torch.long)
        batch_input_mask = torch.tensor(f_pad(1, maxlen), dtype=torch.long)
        batch_segment_ids = torch.tensor(f_pad(2, maxlen), dtype=torch.long)
        batch_confidences = torch.tensor(f_collect(3), dtype=torch.float)
        batch_label_ids = torch.tensor(f_collect(4), dtype=torch.long)
        batch_gcn_vocab_ids_paded = np.array(f_pad2(5, maxlen)).reshape(-1)

        batch_gcn_swop_eye = torch.eye(gcn_vocab_size + 1)[
                                 batch_gcn_vocab_ids_paded
                             ][:, :-1]
        batch_gcn_swop_eye = batch_gcn_swop_eye.view(
            len(batch), -1, gcn_vocab_size
        ).transpose(1, 2)

        return (
            batch_input_ids,
            batch_input_mask,
            batch_segment_ids,
            batch_confidences,
            batch_label_ids,
            batch_gcn_swop_eye,
        )


def compute_tf(corpus):
    tf_dict = {}
    for doc in corpus:
        words = doc
        for word in words:
            if word not in tf_dict:
                tf_dict[word] = 1
            else:
                tf_dict[word] += 1
    return tf_dict


# 计算逆文档频率
def compute_idf(corpus):
    idf_dict = {}
    total_docs = len(corpus)
    for doc in corpus:
        words = set(doc)
        for word in words:
            if word not in idf_dict:
                idf_dict[word] = 1
            else:
                idf_dict[word] += 1
    for word, doc_count in idf_dict.items():
        idf_dict[word] = total_docs / doc_count
    return idf_dict


# 计算 TF-IDF
def compute_tfidf(corpus):
    tf_dict = compute_tf(corpus)
    idf_dict = compute_idf(corpus)

    tfidf_matrix = []
    for doc in corpus:
        tfidf_vector = []
        words = doc
        for word in words:
            tf = tf_dict[word]
            idf = idf_dict[word]
            tfidf = tf * idf
            tfidf_vector.append(tfidf)
        tfidf_matrix.append(tfidf_vector)

    return tfidf_matrix


def normalize_tfidf(tfidf_matrix):
    normalized_matrix = []
    for tfidf_vector in tfidf_matrix:
        norm = np.linalg.norm(tfidf_vector)
        normalized_vector = [tfidf / norm for tfidf in tfidf_vector]
        normalized_matrix.append(normalized_vector)
    return normalized_matrix


def replaced_pfam_mutil(origin_BGC_names, origin_BGC_tokens, top_replacements, y):
    # normalized_tfidf_matrix = normalize_tfidf(compute_tfidf(origin_BGC_tokens))
    freq = [8, 2, 2, 2, 2, 1, 0]

    eda_names = []
    eda_BGC_seq = []
    eda_y = []

    for name, seq, label in zip(origin_BGC_names, origin_BGC_tokens, y):
        for i in range(freq[label]):
            if len(seq) > 10:
                if len(seq) <= 32:
                    top_indices = np.random.randint(0, 5, size=5)
                    replace_indices = np.random.randint(0, len(seq), size=5)
                    new_sentence = seq
                    for ti, ri in zip(top_indices, replace_indices):
                        top = top_replacements.get(ri)
                        new_sentence[ri] = top[ti]
                    eda_BGC_seq.append(new_sentence)
                    eda_y.append(label)
                else:
                    top_indices = np.random.randint(0, 5, size=10)
                    replace_indices = np.random.randint(0, len(seq), size=10)
                    new_sentence = seq
                    for ti, ri in zip(top_indices, replace_indices):
                        top = top_replacements.get(ri)
                        new_sentence[ri] = top[ti]
                    eda_BGC_seq.append(new_sentence)
                    eda_y.append(label)
                eda_names.append(name + f'_EDA_replaced_{i}')

    return eda_names, eda_BGC_seq, eda_y


def find_indices_of_value(arr, target_value):
    indices = []
    for index, value in enumerate(arr):
        if value == target_value:
            indices.append(index)
    return indices


def balance_class(real_token, real_y, eda_token, eda_y):
    all_bgc_token = []
    all_y = []
    count = [0, 0, 0, 0, 0, 0, 0]
    for token, y in zip(real_token, real_y):
        all_bgc_token.append(token)
        all_y.append(y)
        count[y] += 1

    eda_nums = [0, 0, 0, 0, 0, 0, 0]
    for i in range(7):
        if count[i] < 268:
            eda_nums[i] = 268 - count[i]

    for i in range(7):
        if eda_nums[i] > 0:
            indices = find_indices_of_value(eda_y, i)
            random_indices = np.random.randint(0, len(indices), size=eda_nums[i])
            for n in random_indices:
                all_bgc_token.append(eda_token[indices[n]])
                all_y.append(eda_y[indices[n]])

    # shuffle
    combined = list(zip(all_bgc_token, all_y))
    np.random.shuffle(combined)
    shuffled_array1, shuffled_array2 = zip(*combined)

    c = [0, 0, 0, 0, 0, 0, 0]
    for y in shuffled_array2:
        c[y] += 1

    return shuffled_array1, shuffled_array2


def replaced_pfam_binary(origin_seq, top_replacements, labels, num):
    eda_BGC_seq = []
    eda_y = []
    for i in range(num):
        index = random.randint(0, len(origin_seq)-1)
        seq = origin_seq[index]
        if len(seq) > 10:
            if len(seq) <= 32:
                top_indices = np.random.randint(0, 5, size=5)
                replace_indices = np.random.randint(0, len(seq), size=5)
            else:
                top_indices = np.random.randint(0, 5, size=10)
                replace_indices = np.random.randint(0, len(seq), size=10)

            new_sentence = np.array(seq)
            for ti, ri in zip(top_indices, replace_indices):
                top = top_replacements.get(ri)
                new_sentence[ri] = top[ti]
            add = True
            for t in new_sentence:
                if t >= 19450:
                    add = False
                    break
            if add:
                eda_BGC_seq.append(list(new_sentence))
                eda_y.append(labels[index])

    return eda_BGC_seq, eda_y


def eda_binary(seq, top_replacements, labels, num):
    eda_BGC_seq, eda_y = replaced_pfam_binary(seq, top_replacements, labels, num)
    eda_data = {'domain': eda_BGC_seq, 'label': eda_y}
    d = pd.DataFrame(eda_data)
    d.to_csv('add.csv', index=False)

def change(tens):
    max_indeces = torch.argmax(tens, dim=1)
    return max_indeces.tolist()


def count_acc(pred, labels):
    num = [0, 0, 0, 0, 0, 0, 0]
    true = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(pred)):
        if pred[i] == labels[i]:
            true[labels[i]] += 1
        num[labels[i]] += 1
    return num, true


def find_similar_proteins(file_path, pf2domain, vocab, threshold):
    from sklearn.metrics.pairwise import cosine_similarity
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 提取蛋白质名称和嵌入
    protein_names = []
    embeddings = []
    for i, row in df.iterrows():
        name = row[0]
        emb = row[1:].tolist()
        if name in pf2domain.keys():
            protein_names.append(name)
            embeddings.append(emb)

    # 计算余弦相似度
    similarity_matrix = cosine_similarity(embeddings)

    # 初始化结果字典
    similar_proteins = {}

    # 遍历每个蛋白质
    for idx, protein in enumerate(protein_names):
        # 排除自身，找到与其他蛋白质的相似度
        if type(protein) is not str or protein not in pf2domain.keys():
            continue
        similarities = similarity_matrix[idx]
        similar_indices = np.where(similarities > threshold)[0]
        similar_indices = similar_indices[similar_indices != idx]

        # 获取相似度大于阈值的蛋白质名称
        similar_proteins_list = [(vocab[pf2domain[protein_names[i]]], similarities[i]) for i in similar_indices]

        # 按相似度排序并取前3个
        similar_proteins_list = sorted(similar_proteins_list, key=lambda x: x[1], reverse=True)[:3]

        # 保存结果
        similar_proteins[vocab[pf2domain[protein]]] = [p[0] for p in similar_proteins_list]

    return similar_proteins


def get_each_class_idx(train_y):
    a_idx, n_idx, o_idx, p_idx, r_idx, s_idx, t_idx = [], [], [], [], [], [], []
    for i, label in enumerate(train_y):
        if label == 0:
            a_idx.append(i)
        elif label == 1:
            n_idx.append(i)
        elif label == 2:
            o_idx.append(i)
        elif label == 3:
            p_idx.append(i)
        elif label == 4:
            r_idx.append(i)
        elif label == 5:
            s_idx.append(i)
        elif label == 6:
            t_idx.append(i)

    return a_idx, n_idx, o_idx, p_idx, r_idx, s_idx, t_idx
