"""
コード進行トークン化のための語彙定義。

設計方針:
- ルート音(Root)とクオリティ(Quality)を別トークン列として扱う
- コロン区切り表記に対応: "G:maj", "A:min", "B:dim" など
- キーはデータ側でCキーに統一済み
- 拍単位の固定グリッド(4/4拍、1コード=1拍、1区間=8ステップ)
- 語彙にないテンション付きクオリティは QUALITY_ALIASES で基本クオリティに寄せる
"""

# ルート音語彙: クロマチック12音（フラット表記に統一）
ROOT_VOCAB = [
    "PAD",
    "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",
]

# クオリティ語彙（データの表記に合わせてminを使用）
QUALITY_VOCAB = [
    "PAD",
    "maj",   # メジャートライアド
    "min",   # マイナートライアド
    "dim",   # ディミニッシュ
    "aug",   # オーギュメント
    "maj7",  # メジャーセブンス
    "min7",  # マイナーセブンス
    "7",     # ドミナントセブンス
    "hdim7", # ハーフディミニッシュ
    "dim7",  # ディミニッシュセブンス
    "sus2",  # サスペンデッド2nd
    "sus4",  # サスペンデッド4th
]

# 表記ゆれ・テンション付きコード → 語彙内のクオリティ
QUALITY_ALIASES = {
    # 表記ゆれ
    "":      "maj",
    "M":     "maj",
    "m":     "min",
    "M7":    "maj7",
    "m7":    "min7",
    "hdim":  "hdim7",
    "m7b5":  "hdim7",
    "m7-5":  "hdim7",
    "min7b5": "hdim7",
    "o":     "dim",
    "o7":    "dim7",
    "+":     "aug",
    "sus":   "sus4",
    # テンション付き → 基本クオリティ
    "9":     "7",
    "7-9":   "7",
    "7b9":   "7",
    "maj9":  "maj7",
    "M9":    "maj7",
    "min9":  "min7",
    "m9":    "min7",
    "6":     "maj",
    "add9":  "maj",
    "min6":  "min",
    "m6":    "min",
    "maj6":  "maj",
    "M6":    "maj",
    "madd9": "min",
    "minadd9": "min",
    "maj11": "maj7",
    "maj13": "maj7",
    "min11": "min7",
    "min13": "min7",
    "11":    "7",
    "13":    "7",
    "7#9":   "7",
    "7b5":   "7",
    "7#5":   "7",
    "aug7":  "7",
    "7sus4": "sus4",
    "minmaj7": "min",
    "5":     "maj",     # パワーコード（3度なし）はメジャー扱い
}

ROOT_TO_ID    = {tok: i for i, tok in enumerate(ROOT_VOCAB)}
ID_TO_ROOT    = {i: tok for tok, i in ROOT_TO_ID.items()}
QUALITY_TO_ID = {tok: i for i, tok in enumerate(QUALITY_VOCAB)}
ID_TO_QUALITY = {i: tok for tok, i in QUALITY_TO_ID.items()}

ROOT_PAD_ID    = ROOT_TO_ID["PAD"]
QUALITY_PAD_ID = QUALITY_TO_ID["PAD"]

# シャープ→フラット正規化
SHARP_TO_FLAT = {
    "C#": "Db", "D#": "Eb", "E#": "F",
    "F#": "Gb", "G#": "Ab", "A#": "Bb", "B#": "C",
}

# 区切り文字の打ち間違い（"G;maj" など）をコロンに直す
SEPARATOR_TYPOS = [";", "："]

# 区間設計
STEPS_PER_SEGMENT = 8  # 1区間 = 8コード（8拍）


def normalize_quality(quality: str, symbol: str = "") -> str:
    """クオリティ表記を語彙内のトークンに正規化する。"""
    quality = quality.strip()
    if quality in QUALITY_TO_ID:
        return quality
    if quality in QUALITY_ALIASES:
        return QUALITY_ALIASES[quality]
    raise ValueError(
        f"未知のクオリティ '{quality}'（コード: '{symbol}'）\n"
        f"QUALITY_ALIASES に対応を追加してください"
    )


def normalize_root(root: str, symbol: str = "") -> str:
    """ルート音をフラット表記に正規化する。"""
    root = root.strip()
    root = SHARP_TO_FLAT.get(root, root)
    if root not in ROOT_TO_ID:
        raise ValueError(f"未知のルート音 '{root}'（コード: '{symbol}'）")
    return root


def parse_chord_symbol(symbol: str) -> tuple[str, str]:
    """
    コード表記を (root, quality) のペアに分解する。

    対応例:
        "G:maj"  -> ("G",  "maj")
        "A#:min" -> ("Bb", "min")
        "G;maj"  -> ("G",  "maj")    区切りの打ち間違い
        "D:min9" -> ("D",  "min7")   テンションは基本形へ
        "B:hdim" -> ("B",  "hdim7")  表記ゆれ
        "Am"     -> ("A",  "min")
        "Fmaj7"  -> ("F",  "maj7")
    """
    original = symbol
    symbol = symbol.strip()
    for typo in SEPARATOR_TYPOS:
        symbol = symbol.replace(typo, ":")

    # コロン区切り形式
    if ":" in symbol:
        root, quality = symbol.split(":", 1)
        return normalize_root(root, original), normalize_quality(quality, original)

    # 従来形式: "Am", "Fmaj7", "C#m7" など
    # 2文字ルート(Db, C# など)を先に試す
    for n in (2, 1):
        head = symbol[:n]
        if SHARP_TO_FLAT.get(head, head) in ROOT_TO_ID and head != "PAD":
            return (
                normalize_root(head, original),
                normalize_quality(symbol[n:], original),
            )

    raise ValueError(f"ルート音を認識できません: '{original}'")


def check_csv(filepath: str) -> None:
    """
    CSV内の全コードを解析し、正規化の結果と未対応の表記を一覧表示する。
    python vocab.py dataset.csv で実行。
    """
    import csv
    from collections import Counter

    converted = Counter()
    errors = []
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            for i in range(1, STEPS_PER_SEGMENT + 1):
                sym = row[f"Chord{i}"]
                try:
                    root, quality = parse_chord_symbol(sym)
                    if f"{root}:{quality}" != sym.strip():
                        converted[f"{sym.strip()} -> {root}:{quality}"] += 1
                except ValueError as e:
                    errors.append((line_no, str(e).splitlines()[0]))

    print("=== 正規化された表記 ===")
    for k, v in sorted(converted.items()):
        print(f"  {k}  ({v}件)")
    print(f"\n=== 未対応 {len(errors)}件 ===")
    for line_no, msg in errors:
        print(f"  {line_no}行目: {msg}")


if __name__ == "__main__":
    import sys
    check_csv(sys.argv[1] if len(sys.argv) > 1 else "dataset.csv")