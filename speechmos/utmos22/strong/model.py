"""UTMOS strong"""

import torch
from torch import nn, Tensor
import torchaudio # pyright: ignore [reportMissingTypeStubs]

from ..fairseq_alt import Wav2Vec2Model


class UTMOS22Strong(nn.Module):
    """ Saeki_2022 paper's `UTMOS strong learner` inference model (w/o Phoneme encoder)."""

    def __init__(self):
        """Init."""

        super().__init__() # pyright: ignore [reportUnknownMemberType]

        feat_ssl, feat_domain_emb, feat_judge_emb, feat_rnn_h, feat_proj_h = 768, 128, 128, 512, 2048
        feat_cat = feat_ssl + feat_domain_emb + feat_judge_emb

        # SSL/DataDomainEmb/JudgeIdEmb/BLSTM/Projection
        self.wav2vec2 = Wav2Vec2Model()
        self.domain_emb = nn.Parameter(data=torch.empty(1, feat_domain_emb), requires_grad=False)
        self.judge_emb  = nn.Parameter(data=torch.empty(1, feat_judge_emb),  requires_grad=False)
        self.blstm = nn.LSTM(input_size=feat_cat, hidden_size=feat_rnn_h, batch_first=True, bidirectional=True)
        self.projection = nn.Sequential(nn.Linear(feat_rnn_h*2, feat_proj_h), nn.ReLU(), nn.Linear(feat_proj_h, 1))

    def forward(self, wave: Tensor, sr: int) -> Tensor: # pylint: disable=invalid-name
        """wave-to-score :: (B, T) -> (B,) """

        # Resampling :: (B, T) -> (B, T)
        wave = torchaudio.functional.resample(wave, orig_freq=sr, new_freq=16000)

        # Feature extraction :: (B, T) -> (B, Frame, Feat)
        unit_series = self.wav2vec2(wave)
        bsz, frm, _ = unit_series.size()

        # DataDomain/JudgeId Embedding's Batch/Time expansion :: (B=1, Feat) -> (B=bsz, Frame=frm, Feat)
        domain_series = self.domain_emb.unsqueeze(1).expand(bsz, frm, -1)
        judge_series  =  self.judge_emb.unsqueeze(1).expand(bsz, frm, -1)

        # Feature concatenation :: (B, Frame, Feat=f1) + (B, Frame, Feat=f2) + (B, Frame, Feat=f3) -> (B, Frame, Feat=f1+f2+f3)
        cat_series = torch.cat([unit_series, domain_series, judge_series], dim=2)

        # Frame-scale score estimation :: (B, Frame, Feat) -> (B, Frame, Feat) -> (B, Frame, Feat=1) - BLSTM/Projection
        feat_series = self.blstm(cat_series)[0]
        score_series = self.projection(feat_series)

        # Utterance-scale score :: (B, Frame, Feat=1) -> (B, Feat=1) -> (B,) - Time averaging
        utter_score = score_series.mean(dim=1).squeeze(1) * 2 + 3

        return utter_score
