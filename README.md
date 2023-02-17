# Augmenting Adult-Speech Datasets to Generate Synthetic Child-Speech Datasets.

## Abstract

Technologies such as Text-To-Speech (TTS) synthesis and Automatic Speech Recognition (ASR) have become indispensable in providing speech-based Artificial Intelligence (AI) solutions in today’s AI-centric technology sector. However, most of the solutions and research work currently being done focus largely on adult speech. This leads to poor performance of such systems when children’s speech is encountered, resulting in children being unable to benefit from modern speech-related technologies. The main reason for this disparity can be linked to the limited availability of children’s speech datasets that can be used in training modern speech AI systems. In this paper, we propose a speech data augmentation technique to generate synthetic children’s speech from the large amounts of existing adult speech datasets. We use a publicly available Python toolbox for manipulating sound files to tune the pitch and duration of the adult speech utterances to make them sound more child-like. We performed both objective and subjective evaluations on the synthetic child utterances produced to show that adult speech samples were successfully tuned to become more child-like. For the objective evaluation, we compare the similarities of the speaker embeddings before and after the augmentation to a mean child speaker embedding. A Mean Opinion Score (MOS) test was conducted for the subjective evaluation.

## Open-source Code

This repository provides the open-source scripts used for multi-speaker adult audio dataset augmentation, detailed in our paper [Title](link). Our first experimentations augment the [Librispeech](https://www.openslr.org/12/) train-clean-100 dataset, and we use the [CMU kids](https://catalog.ldc.upenn.edu/LDC97S63) dataset for computing child speaker embeddings.<br />

The main functionalities for the augmentation pipeline can be broken down into the following scripts:<br />
- [**Compute_librispeech_cmukids_similarities.py**](https://github.com/C3Imaging/speech-augmentation/blob/main/Compute_librispeech_cmukids_similarities.py): computes the cosine similarity between adult speakers' embedding averaged over all utterances for that speaker to the average child speaker from the CMU kids multi-speaker audio dataset. The [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) library is used here.
- [**forced_alignment_librispeech.py**](https://github.com/C3Imaging/speech-augmentation/blob/main/forced_alignment_librispeech.py): runs forced alignment on Librispeech speakers using wav2vec2.0 ASR model and the Trellis matrix backtracking traversal algorithm to predict timestamps for words in the audio dataset. **NOTE:** transcript files for the audio data are required. The [torchaudio](https://pytorch.org/audio/stable/index.html) library is used here.
- [**cleese_audio_augmentation.py**](https://github.com/C3Imaging/speech-augmentation/blob/main/cleese_audio_augmentation.py): augments the pitch and time duration characteristics of original adult audio data from a multi-speaker dataset. The [CLEESE](https://github.com/neuro-team-femto/cleese) library is used here.

## Installation Requirements (UNIX)

**torch**, **torchaudio** and **resemblyzer** can be installed via **pip install**.

### CLEESE

- To install CLEESE from https://forum.ircam.fr/projects/detail/cleese/ you will be prompted to register for a free account first.
- Once the zip file has been downloaded, unzip it to a convenient /path/to/cleese/
- Then run the following command:
```bash
ln -s /path/to/cleese/cleese-master /usr/local/lib/python3.8/dist-packages/cleese
```
- Python usage:
```python
from cleese.cleese import cleeseProcess
```
- In cleeseProcess.py (source file from the downloaded library), if receiving an import error, change the import statements of cleeseBPF and cleeseEngine to: 
```python
from .cleeseBPF import *
from .cleeseEngine import *
```

## Steps for Reproduction Our Audio Augmentation Experiments

### Step 1: Selecting Suitable Adult Candidate Speakers

We first run the **Compute_librispeech_cmukids_similarities.py** script on Librispeech train-clean-100 dataset + CMU kids dataset. The output produces a text file listing the adult speakers with a cosine similarity score above a specified threshold.

### Step 2: Generate Timestamps for Suitable Adult Speakers' Transcripts

After selecting the suitable adult speakers according to Step 1, we copy the selected speaker folders from Librispeech train-clean-100 to a new folder, so we have a dataset of only the selected speakers. Then we run **forced_alignment_librispeech.py** on that folder. This will populate the new folder with outputted text files with alignments (timestamps) for the words in the transcript files.

### Step 3: Generate Augmented Dataset

Using the adult speakers dataset with time alignments as the input, we run **cleese_audio_augmentation.py** to produce an identically structured dataset of augmented speakers.
