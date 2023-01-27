# Introduction

This Git project provides the open source scripts used for multispeaker adult audio dataset augmentation, detailed in our paper [](). Our first experimentations augment the [Librispeech](https://www.openslr.org/12/) train-clean-100 dataset, and we use the [CMU kids](https://catalog.ldc.upenn.edu/LDC97S63) dataset for computing child speaker embeddings.<br />

The main functionalities for the augmentation pipeline can be broken down into the following scripts:<br />
- **Compute_librispeech_cmukids_similarities.py**: computes the cosine similarity between adult speakers' embedding averaged over all utterances for that speaker to the average child speaker from the CMU kids multispeaker audio dataset. The [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) library is used here.
- **forced_alignment_librispeech.py**: runs forced alignment on Librispeech speakers using wav2vec2.0 ASR model and the Trellis matrix backtracking traversal algorithm to predict timestamps for words in the audio dataset. **NOTE:** transcript files for the audio data are required. The [torchaudio](https://pytorch.org/audio/stable/index.html) library is used here.
- **cleese_audio_augmentation.py**: augments the pitch and time duration characteristics of original adult audio data from a multispeaker dataset. The [CLEESE](https://github.com/neuro-team-femto/cleese) library is used here.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
