"""
交差検証つき学習スクリプト。

1. dataset.csv を曲単位で k 個に分け、検証用を入れ替えながら k 回学習して実力を測る
2. k 回の平均で最も検証誤差が小さかったエポック数を決める
3. 全データでそのエポック数だけ学習し直し、model.pt に保存する

同じ曲の区間は必ず同じ分割に入れる（備考が空欄や Gemini の行は1行ずつ独立扱い）。

CSVの形式:
  Chord1,...,Chord8,sadness,tension,strength,uniqueness[,備考]
"""

import csv
import random
import statistics
import unicodedata
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#　====AIモデル変更時にはここを書き換える=====
from model import LABEL_NAMES, NUM_LABELS, ChordGrandeurLSTM
# =====================================
from vocab import (
    ROOT_TO_ID, QUALITY_TO_ID,
    ROOT_PAD_ID, QUALITY_PAD_ID,
    STEPS_PER_SEGMENT, parse_chord_symbol
)

# ===== ここを必要に応じて変える =====
data_file  = "dataset.csv"
#　====AIモデル変更時にはここを書き換える=====
model_path = "model.pt"
# =====================================
n_folds    = 5       # 分割数
n_epochs   = 10      # 各分割で学習する最大エポック数
batch_size = 8
lr         = 1e-3
seed       = 42
# =====================================

# 曲名として扱わない備考（小文字で比較）
NON_SONG_NOTES = {"", "gemini"}


# ---------- データ ----------

class ChordSegmentDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        root_ids, quality_ids, length, labels = self.samples[idx]
        return (
            torch.tensor(root_ids,    dtype=torch.long),
            torch.tensor(quality_ids, dtype=torch.long),
            torch.tensor(length,      dtype=torch.long),
            torch.tensor(labels,      dtype=torch.float),
        )


def collate_fn(batch):
    root_seqs, qual_seqs, lengths, labels = zip(*batch)
    max_len = max(s.size(0) for s in root_seqs)

    def pad(seqs, pad_val):
        out = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :s.size(0)] = s
        return out

    return (
        pad(root_seqs, ROOT_PAD_ID),
        pad(qual_seqs, QUALITY_PAD_ID),
        torch.stack(lengths),
        torch.stack(labels),
    )


def load_csv(filepath):
    """サンプルのリストと、各サンプルが属するグループ名（曲名）のリストを返す。"""
    samples, groups = [], []
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        note_col = next((c for c in reader.fieldnames if c.startswith("備考")), None)
        for i, row in enumerate(reader):
            root_ids, quality_ids = [], []
            for j in range(1, STEPS_PER_SEGMENT + 1):
                root, quality = parse_chord_symbol(row[f"Chord{j}"])
                root_ids.append(ROOT_TO_ID[root])
                quality_ids.append(QUALITY_TO_ID[quality])
            labels = [float(row[name]) for name in LABEL_NAMES]
            samples.append((root_ids, quality_ids, STEPS_PER_SEGMENT, labels))

            note = (row.get(note_col) or "").strip() if note_col else ""
            groups.append(f"__row{i}" if note.lower() in NON_SONG_NOTES else note)
    return samples, groups


def make_group_folds(groups, k, seed):
    """
    同じグループが同じ分割に入るように、k 個の分割に振り分ける。
    大きいグループから順に、その時点で最も件数の少ない分割へ入れて件数を均等にする。
    戻り値: 各分割の検証用インデックスのリスト
    """
    by_group = defaultdict(list)
    for idx, g in enumerate(groups):
        by_group[g].append(idx)

    group_list = list(by_group.values())
    random.Random(seed).shuffle(group_list)
    group_list.sort(key=len, reverse=True)  # 同じ大きさ同士はシャッフル順を保つ

    folds = [[] for _ in range(k)]
    for idxs in group_list:
        min(folds, key=len).extend(idxs)
    return folds


def make_loader(samples, shuffle):
    return DataLoader(
        ChordSegmentDataset(samples), batch_size=batch_size,
        shuffle=shuffle, collate_fn=collate_fn
    )


# ---------- 学習と評価 ----------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for root_ids, qual_ids, lengths, labels in loader:
        root_ids, qual_ids = root_ids.to(device), qual_ids.to(device)
        lengths, labels    = lengths.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(root_ids, qual_ids, lengths)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    """
    全体のMSE、ラベルごとのMAE、
    予測を整数に丸めた場合のラベルごとのMAE、整数が正解と一致した割合 を返す。
    """
    model.eval()
    sq_err    = torch.zeros(NUM_LABELS)
    abs_err   = torch.zeros(NUM_LABELS)
    r_abs_err = torch.zeros(NUM_LABELS)
    r_hit     = torch.zeros(NUM_LABELS)
    n = 0
    for root_ids, qual_ids, lengths, labels in loader:
        preds = model(root_ids.to(device), qual_ids.to(device), lengths.to(device)).cpu()
        diff  = preds - labels
        sq_err  += (diff ** 2).sum(dim=0)
        abs_err += diff.abs().sum(dim=0)

        rounded = torch.floor(preds + 0.5)   # 四捨五入
        r_diff  = (rounded - labels).abs()
        r_abs_err += r_diff.sum(dim=0)
        r_hit     += (r_diff < 0.5).float().sum(dim=0)
        n += labels.size(0)
    return (
        (sq_err / n).mean().item(),
        (abs_err / n).tolist(),
        (r_abs_err / n).tolist(),
        (r_hit / n).tolist(),
    )


def baseline_mae(train_samples, val_samples):
    """学習データの平均値を常に答えた場合の、ラベルごとの検証MAE。"""
    train_labels = torch.tensor([s[3] for s in train_samples])
    val_labels   = torch.tensor([s[3] for s in val_samples])
    return (val_labels - train_labels.mean(dim=0)).abs().mean(dim=0).tolist()


def new_model(device, fold_seed):
    torch.manual_seed(fold_seed)
    #　====AIモデル変更時にはここを書き換える=====
    model     = ChordGrandeurLSTM().to(device)
    #　====================================
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def run_fold(fold_no, train_samples, val_samples, device):
    """1分割ぶん学習し、エポックごとの (val MSE, ラベル別MAE) を返す。"""
    model, optimizer = new_model(device, seed + fold_no)
    criterion    = nn.MSELoss()
    train_loader = make_loader(train_samples, shuffle=True)
    val_loader   = make_loader(val_samples,   shuffle=False)

    history = []
    for _ in range(n_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        history.append(evaluate(model, val_loader, device))
    return history


# ---------- 表示 ----------

def pad(text, width, align="left"):
    """全角文字を2文字分として、表示幅をそろえる。"""
    text = str(text)
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    space = " " * max(0, width - w)
    return text + space if align == "left" else space + text


def mean_std(values):
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{m:.3f} ± {s:.3f}"


def print_summary(histories, baselines, best_epoch):
    print(f"\n===== 交差検証の結果（{n_folds}分割、エポック {best_epoch} 時点）=====")
    print(pad("ラベル", 12) + pad("モデルMAE", 18, "right") + pad("平均値予測MAE", 20, "right"))
    model_means, base_means = [], []
    for li, name in enumerate(LABEL_NAMES):
        m_vals = [h[best_epoch - 1][1][li] for h in histories]
        b_vals = [b[li] for b in baselines]
        model_means.append(statistics.mean(m_vals))
        base_means.append(statistics.mean(b_vals))
        print(f"{name:<12}{mean_std(m_vals):>18}{mean_std(b_vals):>20}")

    m_all = [statistics.mean(h[best_epoch - 1][1]) for h in histories]
    b_all = [statistics.mean(b) for b in baselines]
    print(pad("平均", 12) + pad(mean_std(m_all), 18, "right") + pad(mean_std(b_all), 20, "right"))

    wins = sum(m < b for m, b in zip(m_all, b_all))
    print(f"\n平均値予測に勝った分割: {wins} / {n_folds}")

    # 予測を整数に丸めた場合との比較
    print("\n----- 予測を整数に丸めた場合 -----")
    print(pad("ラベル", 12) + pad("小数のままMAE", 18, "right")
          + pad("整数に丸めたMAE", 20, "right") + pad("整数の一致率", 16, "right"))
    for li, name in enumerate(LABEL_NAMES):
        m_vals = [h[best_epoch - 1][1][li] for h in histories]
        r_vals = [h[best_epoch - 1][2][li] for h in histories]
        a_vals = [h[best_epoch - 1][3][li] for h in histories]
        print(f"{name:<12}{mean_std(m_vals):>18}{mean_std(r_vals):>20}"
              f"{statistics.mean(a_vals) * 100:>15.1f}%")
    r_all = [statistics.mean(h[best_epoch - 1][2]) for h in histories]
    a_all = [statistics.mean(h[best_epoch - 1][3]) for h in histories]
    print(pad("平均", 12) + pad(mean_std(m_all), 18, "right") + pad(mean_std(r_all), 20, "right")
          + f"{statistics.mean(a_all) * 100:>15.1f}%")
    print("※ ± は分割ごとのばらつき（標準偏差）。ばらつきに比べて差が小さいラベルは、学べているとは言い切れない")


# ---------- メイン ----------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    samples, groups = load_csv(data_file)
    folds = make_group_folds(groups, n_folds, seed)
    print(f"全 {len(samples)} 件（{len(set(groups))} グループ）を {n_folds} 分割: "
          + ", ".join(str(len(f)) for f in folds) + " 件")

    # 1. 交差検証
    histories, baselines = [], []
    for k, val_idx in enumerate(folds, start=1):
        val_set   = set(val_idx)
        train_s   = [s for i, s in enumerate(samples) if i not in val_set]
        val_s     = [samples[i] for i in val_idx]

        history = run_fold(k, train_s, val_s, device)
        base    = baseline_mae(train_s, val_s)
        histories.append(history)
        baselines.append(base)

        best_e = min(range(n_epochs), key=lambda e: history[e][0]) + 1
        best_mae = statistics.mean(history[best_e - 1][1])
        print(f"fold {k}: 最良エポック {best_e:3d}  MAE={best_mae:.3f}  "
              f"（平均値予測 {statistics.mean(base):.3f}）")

    # 2. 全分割の平均で最良のエポック数を決める
    mean_mse = [
        statistics.mean(h[e][0] for h in histories) for e in range(n_epochs)
    ]
    best_epoch = min(range(n_epochs), key=lambda e: mean_mse[e]) + 1
    print_summary(histories, baselines, best_epoch)

    # 3. 全データでそのエポック数だけ学習し直して保存
    print(f"\n全 {len(samples)} 件で {best_epoch} エポック学習して最終モデルを作ります...")
    model, optimizer = new_model(device, seed)
    criterion = nn.MSELoss()
    loader    = make_loader(samples, shuffle=True)
    for _ in range(best_epoch):
        train_one_epoch(model, loader, optimizer, criterion, device)
    torch.save(model.state_dict(), model_path)
    print(f"最終モデルを {model_path} に保存しました")