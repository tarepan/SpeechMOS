<div align="center">

# 🎧 SpeechMOS 🎧 <!-- omit in toc -->

</div>

Predict subjective speech score with only 2 lines of code, with various MOS prediction systems.

```python
predictor = torch.hub.load("tarepan/SpeechMOS:main", "utmos22_strong", trust_repo=True)
score = predictor(wave)
# xx, good quality speech!
```

## Demo
Predict naturalness (Naturalness Mean-Opinion-Score) of your audio by UTMOS:  

```python
import torch
import librosa

wave, sr = librosa.load("<your_audio>.wav", sr=None, mono=True)
predictor = torch.hub.load("tarepan/SpeechMOS:main", "utmos22_strong", trust_repo=True)
score = predictor(torch.from_numpy(wave).unsqueeze(0))
# 
```

## How to Use
SpeechMOS use `torch.hub` built-in model loader, so no needs of library import😉  

First, instantiate a MOS predictor with model specifier string:
```python
predictor = torch.hub.load("tarepan/SpeechMOS:main", "<model_specifier>", trust_repo=True)
```

Then, pass tensor of speeches :: `(Batch, Time)`:
```python
waves_tensor = torch.rand((2, 16000)) # Two speeches, 16,000 samples per speech
score = predictor(waves_tensor)
#
```

Returned scores :: `(Batch,)` are each speech's predicted MOS.  
If you hope MOS average over speeches (e.g. for TTS model evaluation), just average them:
```python
average_score = score.mean().item()
#
```

## Predictors
This repository is reimplementation collection of various MOS prediction systems.  
Currently we provide below models:  

| Model        | specifier        | paper                         |
|--------------|------------------|-------------------------------|
| UTMOS strong | `utmos22_strong` | [Saeki (2022)][paper_utmos22] |


### Acknowlegements <!-- omit in toc -->
- UTMOS
  - [paper][paper_utmos22]
  - [repository](https://github.com/sarulab-speech/UTMOS22)


[paper_utmos22]: https://arxiv.org/abs/2204.02152
