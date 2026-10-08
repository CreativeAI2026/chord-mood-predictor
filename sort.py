
"""
指定した雰囲気に近いコード進行を探して、近い順に表示するスクリプト。

各ラベル(悲しさ・緊張感・力強さ・独自性)について 1〜5 の目標値を指定すると、
候補CSVのコード進行を学習済みモデルで採点し、目標値との距離が小さい順に並べる。
目標値を指定しなかったラベルは「どれでもよい」として距離の計算から外す。

入力: 候補CSV（Chord1〜Chord8 の列があればよい。ラベル列はあっても無視）
出力: 画面表示 + sort_output.csv
"""

import csv
import math

import torch

from model import ChordGrandeurLSTM, LABEL_NAMES, SCORE_MIN, SCORE_MAX
from vocab import STEPS_PER_SEGMENT, parse_chord_symbol
from predict import load_input_csv, predict, pad, LABEL_DISPLAY

# ===== ここを必要に応じて変える =====
# 候補にするコード進行のCSV（複数指定可）
candidate_files = ["predict_input.csv"]
model_path      = "model.pt"
output_file     = "sort_output.csv"
top_n           = 20      # 表示する件数

# True: 実行時に目標値をキーボードで入力する
# False: 下の TARGET をそのまま使う
ask_input = True

# ask_input = False のときに使う目標値（None は指定なし）
TARGET = {
    "sadness":    4,
    "tension":    None,
    "strength":   2,
    "uniqueness": None,
}
# =====================================


def ask_targets():
    """ラベルごとに 1〜5 の目標値を入力してもらう。空欄なら指定なし。"""
    print(f"各項目の目標値を {SCORE_MIN:g}〜{SCORE_MAX:g} で入力してください（Enterのみで指定なし）")
    targets = {}
    for name in LABEL_NAMES:
        label = LABEL_DISPLAY.get(name, name)
        while True:
            text = input(f"  {label}: ").strip()
            if text == "":
                targets[name] = None
                break
            try:
                value = float(text)
            except ValueError:
                print("    数値を入力してください")
                continue
            if SCORE_MIN <= value <= SCORE_MAX:
                targets[name] = value
                break
            print(f"    {SCORE_MIN:g}〜{SCORE_MAX:g} の範囲で入力してください")
    return targets


def load_candidates(filepaths):
    """
    複数の候補CSVを読み込み、同じコード進行（表記ゆれを正規化したうえで同一のもの）を1件にまとめる。
    """
    samples, rows, seen = [], [], set()
    for path in filepaths:
        file_samples, file_rows = load_input_csv(path)
        for sample, row in zip(file_samples, file_rows):
            key = tuple(
                parse_chord_symbol(row[f"Chord{j}"])
                for j in range(1, STEPS_PER_SEGMENT + 1)
            )
            if key in seen:
                continue
            seen.add(key)
            samples.append(sample)
            rows.append({f"Chord{j}": row[f"Chord{j}"] for j in range(1, STEPS_PER_SEGMENT + 1)})
    return samples, rows


def distance(scores, targets):
    """指定されたラベルだけで、目標値とのユークリッド距離を計算する。"""
    diffs = [
        (score - targets[name]) ** 2
        for name, score in zip(LABEL_NAMES, scores)
        if targets.get(name) is not None
    ]
    return math.sqrt(sum(diffs))


def rank(rows, scores, targets):
    """距離の小さい順に (距離, 行, スコア) を並べて返す。"""
    ranked = [(distance(s, targets), row, s) for row, s in zip(rows, scores)]
    ranked.sort(key=lambda x: x[0])
    return ranked


def print_ranking(ranked, targets, n):
    chord_w, score_w = 72, 8
    headers = [LABEL_DISPLAY.get(name, name) for name in LABEL_NAMES]

    target_text = "  ".join(
        f"{LABEL_DISPLAY.get(name, name)}={targets[name]:g}"
        for name in LABEL_NAMES if targets.get(name) is not None
    )
    print(f"\n目標: {target_text}")
    print(
        pad("順位", 6) + pad("コード進行", chord_w)
        + "".join(pad(h, score_w, "right") for h in headers)
        + pad("距離", score_w, "right")
    )
    print("-" * (6 + chord_w + score_w * (len(LABEL_NAMES) + 1)))
    for i, (dist, row, s) in enumerate(ranked[:n], start=1):
        chords = " ".join(row[f"Chord{j}"] for j in range(1, STEPS_PER_SEGMENT + 1))
        print(
            pad(i, 6) + pad(chords, chord_w)
            + "".join(pad(v, score_w, "right") for v in s)
            + pad(f"{dist:.2f}", score_w, "right")
        )


def save_ranking(filepath, ranked):
    fieldnames = (
        ["rank"] + [f"Chord{j}" for j in range(1, STEPS_PER_SEGMENT + 1)]
        + LABEL_NAMES + ["distance"]
    )
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (dist, row, s) in enumerate(ranked, start=1):
            out = {"rank": i, **row, "distance": round(dist, 3)}
            out.update(dict(zip(LABEL_NAMES, s)))
            writer.writerow(out)
    print(f"\n全 {len(ranked)} 件の順位を {filepath} に保存しました")


if __name__ == "__main__":
    targets = ask_targets() if ask_input else dict(TARGET)
    if all(v is None for v in targets.values()):
        print("目標値が1つも指定されていないため終了します")
        raise SystemExit

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChordGrandeurLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    samples, rows = load_candidates(candidate_files)
    print(f"\n{len(samples)} 件の候補を採点しています...")
    scores = predict(model, samples, device)

    ranked = rank(rows, scores, targets)
    print_ranking(ranked, targets, top_n)
    save_ranking(output_file, ranked)
