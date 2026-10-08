"""
コード進行の「悲しさ・緊張感・力強さ」を同時予測するLSTM回帰モデル。

入力:
  root_ids    : ルート音トークン列 [batch, seq_len=8]
  quality_ids : クオリティトークン列 [batch, seq_len=8]
出力:
  scores      : [batch, 3]  (悲しさ, 緊張感, 力強さ) 各1.0〜5.0
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from vocab import ROOT_VOCAB, QUALITY_VOCAB, ROOT_PAD_ID, QUALITY_PAD_ID

# 出力ラベルの定義
LABEL_NAMES = ["sadness", "tension","uniqueness"]
NUM_LABELS  = len(LABEL_NAMES)
SCORE_MIN   = 1.0
SCORE_MAX   = 5.0


class ChordGrandeurLSTM(nn.Module):
    def __init__(
        self,
        root_vocab_size: int    = len(ROOT_VOCAB),
        quality_vocab_size: int = len(QUALITY_VOCAB),
        root_emb_dim: int       = 16,
        quality_emb_dim: int    = 16,
        hidden_dim: int         = 64,
        num_layers: int         = 2,
        dropout: float          = 0.3,
        bidirectional: bool     = True,
    ):
        super().__init__()

        self.root_embedding = nn.Embedding(
            root_vocab_size, root_emb_dim, padding_idx=ROOT_PAD_ID
        )
        self.quality_embedding = nn.Embedding(
            quality_vocab_size, quality_emb_dim, padding_idx=QUALITY_PAD_ID
        )

        input_dim = root_emb_dim + quality_emb_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        num_directions = 2 if bidirectional else 1
        lstm_out_dim   = hidden_dim * num_directions

        # 3ラベルを同時に出力する回帰ヘッド
        self.regressor = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_LABELS),  # 出力3つ
        )

    def forward(self, root_ids, quality_ids, lengths):
        """
        Returns:
            scores: [batch, 3]
                    dim0=sadness, dim1=tension, dim2=strength
                    各値は SCORE_MIN〜SCORE_MAX (1.0〜5.0)
        """
        root_emb = self.root_embedding(root_ids)
        qual_emb = self.quality_embedding(quality_ids)
        x = torch.cat([root_emb, qual_emb], dim=-1)

        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)

        num_directions = 2 if self.lstm.bidirectional else 1
        h_n = h_n.view(
            self.lstm.num_layers, num_directions, h_n.size(1), self.lstm.hidden_size
        )
        last = h_n[-1]
        final_hidden = (
            torch.cat([last[0], last[1]], dim=-1) if num_directions == 2 else last[0]
        )

        raw    = self.regressor(final_hidden)                          # [batch, 3]
        scores = SCORE_MIN + torch.sigmoid(raw) * (SCORE_MAX - SCORE_MIN)
        return scores


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 4, 8

    model       = ChordGrandeurLSTM()
    root_ids    = torch.randint(1, len(ROOT_VOCAB),    (B, T))
    quality_ids = torch.randint(1, len(QUALITY_VOCAB), (B, T))
    lengths     = torch.tensor([8, 8, 8, 8])

    scores = model(root_ids, quality_ids, lengths)
    print("scores shape:", scores.shape)   # [4, 3]
    print("scores:\n", scores)
    assert scores.shape == (B, NUM_LABELS)
    assert torch.all(scores >= SCORE_MIN) and torch.all(scores <= SCORE_MAX)
    print("OK: 出力は1〜5の範囲に収まっています")