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
- If receiving an import error, then:
1. In /path/to/cleese/cleese-master/cleese/__init__.py, change import statement to:
```python
from .cleeseProcess import *
```
2. In /path/to/cleese/cleese-master/cleese/cleeseProcess.py, change the import statements of cleeseBPF and cleeseEngine to: 
```python
from .cleeseBPF import *
from .cleeseEngine import *
```

## Steps for Reproduction Our Audio Augmentation Experiments

### Step 1: Selecting Suitable Adult Candidate Speakers

We first run the **Compute_librispeech_cmukids_similarities.py** script on Librispeech train-clean-100 dataset + CMU kids dataset. The output results in a new folder that contains copied Librispeech speaker folders with a cosine similarity score above a specified threshold, appended with their gender tag.

### Step 2: Generate Timestamps for Suitable Adult Speakers' Transcripts

After selecting the suitable adult speakers according to Step 1, we run **forced_alignment_librispeech.py** on the new folder. This will populate the speaker folders with a subfolder that contains text files with time alignments (timestamps for the words in each transcript) for each audio file.

### Step 3: Generate Augmented Dataset

Using the dataset from Step 2, we run **cleese_audio_augmentation.py** to produce a new identically structured dataset of augmented speakers. The CLEESE configuration used for augmentation experiments can be found in the file **cleeseConfig_all_lj.py**

# Other Important Scripts for your convenience

## ASR Inference
Current ASR models available:
- wav2vec2 (fairseq framework): You can run `wav2vec2_infer_custom.py` (run `python wav2vec2_infer_custom.py --help` for a description of the usage) to generate hypothesis text transcripts from an unlabelled audio dataset.
**NOTE:** You must manually specify the ASR pipeline you want to use in the code under the comment `# create model + decoder pair`. There are a number of possible combinations of ASR model + Decoder + [optional external Language Model] to choose from. Please see `decoding_utils/Wav2Vec2_Decoder_Factory` class for a list of the implemented pipelines.<br />
**Wav2Vec2 ASR Pipeline Notes:**
  1. a wav2vec2 checkpoint file can have its arch defined in the **"args"** field or **"cfg"** field, depending on the version of the fairseq framework used to train the model. If you get an error using a "get_args_" function from the Wav2Vec2 factory class, the checkpoint is likely a **"cfg"** one, thus try using the "get_cfg_" equivalent function instead and vice versa).
  2. You must also manually edit the file `Tools/decoding_utils.py` with paths local to you in the following places:<br />
- `BeamSearchKenLMDecoder_Fairseq -> __init__ -> decoder_args -> kenlm_model` (if using a KenLM external language model)
- `BeamSearchKenLMDecoder_Fairseq -> __init__ -> decoder_args -> lexicon` (if using a KenLM external language model)
- `TransformerDecoder -> __init__ -> transformerLM_root_folder` (if using a TransformerLM external language model)
- `TransformerDecoder -> __init__ -> lexicon` (if using a TransformerLM external language model)
<br /><br />
Future ASR models to be integrated with their own factories:<br />
- Conformer-Transducer (NeMo framework)

## Forced Alignment

Forced alignment between paired text and audio data (a.k.a generating timestaps for the words in the text transcript) can be performed using ASR or TTS models.

Current ASR models available:
- wav2vec2 (from torchaudio **AND** custom wav2vec2 models now supported): You can run `wav2vec2_forced_alignment_libri.py` (run `python wav2vec2_forced_alignment_libri.py --help` for a description of the usage) to generate time alignments for paired <text,audio> datasets whose transcripts are saved in LibriSpeech **OR** LibriTTS format.
- whisper: You can generate transcripts with Whisper and time align the generated transcript with the speech file using Dynamic Time Warping (run `python whisper_forced_alignment.py --help` for a description of the usage).

## Speaker Diarization

Current Speaker Diarization models available:
- Pyannote
- Resemblyzer
- NeMo MSDD<br /><br />
All are accessed using `diarization/main.py` (run `python diarization/main.py --help` for a description of the usage).<br />
The configuration for a diarization run can be modified in the `diarization/diarization.yaml` file.<br /><br />
**Resemblyzer Diairization Notes:**
  1. Resemblyzer allows the user to specify an example speech audio wav for each speaker from which a speaker embedding will be created, across all multipeaker audiofiles. To enable this functionality, you must create the following folder structure:<br /><br />
     In the main folder where the multispeaker audiofiles are located, create a `speaker-samples` subfolder. In it, create a subfolder for each audiofile with the same name as the audiofile. In each of these subfolders, have a `<SPEAKER_ID>.WAV` file for each speaker you wish to segment in the corresponding multispeaker audiofile from the main folder.<br /><br />
**NOTE:** Resemblyzer **requires** separate audiofiles from which to create speaker embeddings, so this step is **mandatory**.


