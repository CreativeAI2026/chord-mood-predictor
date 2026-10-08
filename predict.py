"""
学習済みモデルで新しいコード進行の各スコアを予測するスクリプト。
予測するラベルは model.py の LABEL_NAMES に従う
（現在は 悲しさ・緊張感・力強さ・独自性 の4つ）。

入力CSV列: Chord1, Chord2, ..., Chord8  （labelなし）
出力CSV列: Chord1〜Chord8 + LABEL_NAMES の各列
"""

import csv
import unicodedata

import torch

from model import ChordGrandeurLSTM, LABEL_NAMES
from vocab import (
    ROOT_TO_ID, QUALITY_TO_ID,
    STEPS_PER_SEGMENT, parse_chord_symbol
)

# ===== ここを必要に応じて変える =====
input_file  = "predict_input.csv"
output_file = "predict_output.csv"
model_path  = "model.pt"
as_integer  = True    # True: スコアを四捨五入して整数で出す / False: 小数第1位まで出す
# =====================================

# 画面表示用の日本語ラベル名（LABEL_NAMES にない名前は英語のまま表示）
LABEL_DISPLAY = {
    "sadness":    "悲しさ",
    "tension":    "緊張感",
    "strength":   "力強さ",
    "uniqueness": "独自性",
}


def load_input_csv(filepath):
    samples = []
    rows    = []
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chord_keys  = [f"Chord{i}" for i in range(1, STEPS_PER_SEGMENT + 1)]
            root_ids    = []
            quality_ids = []
            for key in chord_keys:
                root, quality = parse_chord_symbol(row[key])
                root_ids.append(ROOT_TO_ID[root])
                quality_ids.append(QUALITY_TO_ID[quality])
            samples.append((root_ids, quality_ids))
            rows.append(row)
    return samples, rows


def predict(model, samples, device, as_integer=False):
    """
    各コード進行のスコアを予測する。
    as_integer=True なら四捨五入した整数、False なら小数第1位までの値を返す。
    """
    model.eval()
    results = []
    with torch.no_grad():
        for root_ids, quality_ids in samples:
            root_t  = torch.tensor([root_ids],    dtype=torch.long).to(device)
            qual_t  = torch.tensor([quality_ids], dtype=torch.long).to(device)
            lengths = torch.tensor([STEPS_PER_SEGMENT], dtype=torch.long).to(device)
            scores  = model(root_t, qual_t, lengths)[0]  # [NUM_LABELS]
            if as_integer:
                results.append([int(s.item() + 0.5) for s in scores])
            else:
                results.append([round(s.item(), 1) for s in scores])
    return results


def save_output_csv(filepath, rows, scores):
    if not rows:
        print("データが空です")
        return
    fieldnames = list(rows[0].keys()) + LABEL_NAMES
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, score_list in zip(rows, scores):
            for name, val in zip(LABEL_NAMES, score_list):
                row[name] = val
            writer.writerow(row)
    print(f"結果を {filepath} に保存しました")


def display_width(text):
    """全角文字を2、半角文字を1として表示幅を数える。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(text))


def pad(text, width, align="left"):
    """表示幅を考慮して文字列を指定幅にそろえる。"""
    text  = str(text)
    space = " " * max(0, width - display_width(text))
    return text + space if align == "left" else space + text


def print_results(rows, scores):
    chord_w = 72  # 最長の "Gb:hdim7" ×8 でも収まる幅
    score_w = 8
    headers = [LABEL_DISPLAY.get(name, name) for name in LABEL_NAMES]

    print()
    print(pad("コード進行", chord_w) + "".join(pad(h, score_w, "right") for h in headers))
    print("-" * (chord_w + score_w * len(LABEL_NAMES)))
    for row, score_list in zip(rows, scores):
        chords = " ".join(row[f"Chord{j}"] for j in range(1, STEPS_PER_SEGMENT + 1))
        print(pad(chords, chord_w) + "".join(pad(s, score_w, "right") for s in score_list))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = ChordGrandeurLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"モデルを {model_path} から読み込みました")

    samples, rows = load_input_csv(input_file)
    print(f"{len(samples)} 件のコード進行を予測します...")

    scores = predict(model, samples, device, as_integer=as_integer)
    print_results(rows, scores)
    save_output_csv(output_file, rows, scores)