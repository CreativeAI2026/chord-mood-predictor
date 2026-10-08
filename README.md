# chord-mood-predictor

8 コードのコード進行から「悲しさ」「緊張感」「独自性」を 1〜5 で予測する LSTM モデルです。予測結果を使って、指定した雰囲気に近いコード進行を探すこともできます。

## セットアップ

### リポジトリを取得

```bash
git clone https://github.com/CreativeAI2026/chord-mood-predictor.git
cd chord-mood-predictor
```

以降のコマンドはすべてこのフォルダで実行します。

### Python 3.13 をインストール

macOS（[Homebrew](https://brew.sh/ja/) が必要）:

```bash
brew install python@3.13
```

Windows（PowerShell）:

```powershell
winget install Python.Python.3.13
```

インストール後、ターミナルを開き直してください。

### ライブラリをインストール

macOS:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows（PowerShell）:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 実行

学習済みモデル `model.pt` が入っているので、すぐに試せます。

### コード進行の雰囲気を予測

```bash
python predict.py
```

`predict_input.csv` の各コード進行（1 行に 8 コード）を採点し、結果を `predict_output.csv` に保存します。

### 目標の雰囲気に近いコード進行を探す

```bash
python sort.py
```

項目ごとに 1〜5 の目標値を聞かれます（気にしない項目は Enter のみ）。

目標に近い順に上位 20 件を表示し、全件の順位を `sort_output.csv` に保存します。

### モデルを学習し直す

```bash
python train.py
```

`dataset.csv`（コード進行ごとに各項目を 1〜5 で評価したデータ）で学習し、`model.pt` を上書きします。最初に交差検証で精度（予測のずれ MAE）を表示し、そのあと全データで学習します。

### コード進行を音で確認

`chord-player.html` をブラウザで開き、「別のCSVを読み込む」から CSV を開くと再生できます。
