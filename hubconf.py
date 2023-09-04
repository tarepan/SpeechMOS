"""torch.hub configuration."""

dependencies = ["torch", "torchaudio"]

import torch                                             # pylint: disable=wrong-import-position

from speechmos.utmos22.strong.model import UTMOS22Strong # pylint: disable=wrong-import-position


URLS = {
    "utmos22_strong": "https://github.com/tarepan/SpeechMOS/releases/download/v0.0.0/utmos22_strong_step7459.pt",
}
# [Origin]
# "utmos22_strong" is derived from official sarulab-speech/UTMOS22 'UTMOS strong learner' checkpoint, under MIT lisence (Copyright 2022 Saruwatari&Koyama laboratory, The University of Tokyo, https://github.com/sarulab-speech/UTMOS22/blob/master/LICENSE).
# Weight transfer code is in my fork (`/demo/utmos_strong_alt`).


def utmos22_strong(progress: bool = True) -> UTMOS22Strong:
    """
    `UTMOS strong learner` speech naturalness MOS predictor.

    Args:
        progress - Whether to show model checkpoint load progress
    """

    state_dict = torch.hub.load_state_dict_from_url(url=URLS["utmos22_strong"], map_location="cpu", progress=progress)
    model = UTMOS22Strong()
    model.load_state_dict(state_dict)
    model.eval()

    return model
