# Introduction

This Git project provides the open source scripts used for multispeaker adult audio dataset augmentation, detailed in our paper [](). Our first experimentations augment the [Librispeech](https://www.openslr.org/12/) train-clean-100 dataset, and we use the [CMU kids](https://catalog.ldc.upenn.edu/LDC97S63) dataset for computing child speaker embeddings.<br />

The main functionalities for the augmentation pipeline can be broken down into the following scripts:<br />
- **Compute_librispeech_cmukids_similarities.py**: computes the cosine similarity between adult speakers' embedding averaged over all utterances for that speaker to the average child speaker from the CMU kids multispeaker audio dataset. The [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) library is used here.
- **forced_alignment_librispeech.py**: runs forced alignment on Librispeech speakers using wav2vec2.0 ASR model and the Trellis matrix backtracking traversal algorithm to predict timestamps for words in the audio dataset. **NOTE:** transcript files for the audio data are required. The [torchaudio](https://pytorch.org/audio/stable/index.html) library is used here.
- **cleese_audio_augmentation.py**: augments the pitch and time duration characteristics of original adult audio data from a multispeaker dataset. The [CLEESE](https://github.com/neuro-team-femto/cleese) library is used here.

## Audio Augmentation Pipeline

### Step 1: Selecting Suitable Adult Candidate Speakers
We first run the **Compute_librispeech_cmukids_similarities.py** script on Librispeech train-clean-100 dataset + CMU kids dataset. The output produces a text file listing the adult speakers with a cosine similarity score above a specified threshold.

### Step 2: Generate Timestamps for Suitable Adult Speakers' Transcripts
After selecting the suitable adult speakers according to Step 1, we copy the selected speaker folders from Librispeech train-clean-100 to a new folder, so we have a dataset of only the selected speakers. Then we run **forced_alignment_librispeech.py** on that folder. This will populate the new folder with outputted text files with alignments (timestamps) for the words in the transcript files.

### Step 3: Generate Augmented Dataset
Using the adult speakers dataset with time alignments as the input, we run **cleese_audio_augmentation.py** to produce an identically structured dataset of augmented speakers.
