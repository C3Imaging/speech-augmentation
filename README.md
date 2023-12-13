# Augmenting Adult-Speech Datasets to Generate Synthetic Child-Speech Datasets.

## Abstract

Technologies such as Text-To-Speech (TTS) synthesis and Automatic Speech Recognition (ASR) have become indispensable in providing speech-based Artificial Intelligence (AI) solutions in today’s AI-centric technology sector. However, most of the solutions and research work currently being done focus largely on adult speech. This leads to poor performance of such systems when children’s speech is encountered, resulting in children being unable to benefit from modern speech-related technologies. The main reason for this disparity can be linked to the limited availability of children’s speech datasets that can be used in training modern speech AI systems. In this paper, we propose a speech data augmentation technique to generate synthetic children’s speech from the large amounts of existing adult speech datasets. We use a publicly available Python toolbox for manipulating sound files to tune the pitch and duration of the adult speech utterances to make them sound more child-like. We performed both objective and subjective evaluations on the synthetic child utterances produced to show that adult speech samples were successfully tuned to become more child-like. For the objective evaluation, we compare the similarities of the speaker embeddings before and after the augmentation to a mean child speaker embedding. A Mean Opinion Score (MOS) test was conducted for the subjective evaluation.

## Open-source Code

This repository provides the open-source scripts used for multi-speaker adult audio dataset augmentation, detailed in our paper [Title](link). Our first experimentations augment the [Librispeech](https://www.openslr.org/12/) train-clean-100 dataset, and we use the [CMU kids](https://catalog.ldc.upenn.edu/LDC97S63) dataset for computing child speaker embeddings.<br />

The main functionalities for the augmentation pipeline can be broken down into the following scripts:<br />
- [**Compute_librispeech_cmukids_similarities.py**](https://github.com/C3Imaging/speech-augmentation/blob/main/Compute_librispeech_cmukids_similarities.py): computes the cosine similarity between adult speakers' embedding averaged over all utterances for that speaker to the average child speaker from the CMU kids multi-speaker audio dataset. The [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) library is used here.
- [**forced_alignment_librispeech.py**](https://github.com/C3Imaging/speech-augmentation/blob/main/wav2vec2_forced_alignment_libri.py): runs forced alignment on Librispeech speakers using wav2vec2.0 ASR model and the Trellis matrix backtracking traversal algorithm to predict timestamps for words in the audio dataset. **NOTE:** transcript files for the audio data are required. The [torchaudio](https://pytorch.org/audio/stable/index.html) library is used here.
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

After selecting the suitable adult speakers according to Step 1, we run **wav2vec2_forced_alignment_libri.py** on the new folder. This will populate the speaker folders with a subfolder that contains text files with time alignments (timestamps for the words in each transcript) for each audio file.

### Step 3: Generate Augmented Dataset

Using the dataset from Step 2, we run **cleese_audio_augmentation.py** to produce a new identically structured dataset of augmented speakers. The CLEESE configuration used for augmentation experiments can be found in the file **cleeseConfig_all_lj.py**

# Other Important Scripts for your convenience

## ASR Inference
Current ASR models available:
- wav2vec2 ([fairseq](https://github.com/facebookresearch/fairseq) framework): You can run `wav2vec2_infer_custom.py` to generate hypothesis text transcripts from an unlabelled audio dataset.<br />
Run `python wav2vec2_infer_custom.py --help` for a description of the usage.<br /><br />
**NOTE:** You must manually specify the ASR pipeline you want to use in the code under the comment `# create model + decoder pair`. There are a number of possible combinations of ASR model + Decoder + [optional external Language Model] to choose from. Please see `decoding_utils/Wav2Vec2_Decoder_Factory` class for a list of the implemented pipelines.<br /><br />
**Wav2Vec2 ASR Pipeline Notes:**
  1. a wav2vec2 checkpoint file can have its architecture defined in the **"args"** field or **"cfg"** field, depending on the version of the fairseq framework used to train the model. If you get an error using a "get_args_" function from the Wav2Vec2 factory class, the checkpoint is likely a **"cfg"** one, thus try using the "get_cfg_" equivalent function instead and vice versa).
  2. You must also manually edit the file `Tools/decoding_utils.py` with paths local to you in the following places:<br />
  - `BeamSearchKenLMDecoder_Fairseq -> __init__ -> decoder_args -> kenlm_model` (if using a KenLM external language model)
  - `BeamSearchKenLMDecoder_Fairseq -> __init__ -> decoder_args -> lexicon` (if using a KenLM external language model)
  - `TransformerDecoder -> __init__ -> transformerLM_root_folder` (if using a TransformerLM external language model)
  - `TransformerDecoder -> __init__ -> lexicon` (if using a TransformerLM external language model)<br /><br />

  **UPDATE:** word-level time alignment output information, as well as multiple hypotheses output is now supported. Please read [Time Aligned Predictions and Forced Alignment](https://github.com/C3Imaging/speech-augmentation?tab=readme-ov-file#time-aligned-predictions-and-forced-alignment) section for more details.<br />
- Whisper ([whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) framework): You can run `whisper_time_alignment.py` to generate hypothesis text transcripts from an unlabelled audio dataset with optional word-level time alignment output information as well as multiple hypotheses output.<br />
Run `python whisper_time_alignment.py --help` for a description of the usage.<br />
Please read [Time Aligned Predictions and Forced Alignment](https://github.com/C3Imaging/speech-augmentation?tab=readme-ov-file#time-aligned-predictions-and-forced-alignment) section for more details.<br /><br />
- Conformer ([NeMo](https://github.com/NVIDIA/NeMo) framework): Please see the [NeMo ASR Experiments](https://github.com/abarcovschi/nemo_asr/blob/main/README.md) project for more details.

## Time Aligned Predictions and Forced Alignment

Generating time alignment information for predictions is possible. Also, forced alignment between paired text and audio data (a.k.a aligning **known** text transcript with audio file) can be performed using ASR or TTS models.

Current available ASR-based approaches:
- **Wav2Vec2 inference with time alignment:** The script `wav2vec2_infer_custom.py` can be used to create word-level time aligned transcripts using wav2vec2 models **both** from the torchaudio library or from a custom trained/finetuned checkpoint using the [fairseq framework](https://github.com/facebookresearch/fairseq).<br /><br />
Run `python wav2vec2_infer_custom.py --help` for a description of the usage.<br /><br />
There are multiple ASR model + Decoder options available:
  - Torchaudio Wav2Vec2 model + Greedy Decoder from torchaudio (multiple hypotheses and word-level timestamps unavailable);
  - Torchaudio Wav2Vec2 model + lexicon-free Beam Search Decoder without external language model from torchaudio (multiple hypotheses and word-level timestamps available);
  - Torchaudio Wav2Vec2 model + lexicon-based Beam Search Decoder with KenLM external language model from torchaudio (multiple hypotheses and word-level timestamps available);
  - Wav2Vec2 model checkpoint trained using fairseq framework + Viterbi Decoder from fairseq (multiple hypotheses unavailable, word-level timestamps available);
  - Wav2Vec2 model checkpoint trained using fairseq framework + Greedy Decoder from torchaudio (multiple hypotheses and word-level timestamps unavailable);
  - Wav2Vec2 model checkpoint trained using fairseq framework + lexicon-free Beam Search Decoder without external language model from torchaudio (multiple hypotheses and word-level timestamps available);
  - Wav2Vec2 model checkpoint trained using fairseq framework + lexicon-based Beam Search Decoder with KenLM external language model from torchaudio (multiple hypotheses and word-level timestamps available);
  - Wav2Vec2 model checkpoint trained using fairseq framework + lexicon-based Beam Search Decoder with KenLM external language model from fairseq (multiple hypotheses and word-level timestamps available);
  - Wav2Vec2 model checkpoint trained using fairseq framework + lexicon-based Beam Search Decoder with neural Transformer-based external language model from fairseq (multiple hypotheses and word-level timestamps available).<br /><br />
 **NOTE:** By changing the source code of [fairseq/examples/speech_recognition/w2l_decoder.py](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_recognition/w2l_decoder.py) to that in [fairseq forked](https://github.com/abarcovschi/fairseq-fork/blob/main/examples/speech_recognition/w2l_decoder.py), word-level timestamps can be returned from **all** the decoders in this file (W2lViterbiDecoder, W2lKenLMDecoder, W2lFairseqLMDecoder).<br /><br />
- **Whisper inference with time alignment:** You can generate transcripts with Whisper and time align the generated transcript with the speech file using Dynamic Time Warping.<br />
Run `python whisper_time_alignment.py --help` for a description of the usage).<br /><br />
**UPDATE:** By changing the source code of [openai/whisper/decoding.py](https://github.com/openai/whisper/blob/main/whisper/decoding.py), [openai/whisper/transcribe.py](https://github.com/openai/whisper/blob/main/whisper/transcribe.py) and [linto-ai/whisper-timestamped/transcribe.py](https://github.com/linto-ai/whisper-timestamped/blob/master/whisper_timestamped/transcribe.py) to those in the [openai/whisper forked](https://github.com/abarcovschi/whisper-fork) and [linto-ai/whisper-timestamped forked](https://github.com/abarcovschi/whisper-timestamped-fork) repositories, multiple hypotheses can be returned from beam search decoding, instead of the default best hypothesis offered by the original projects.<br /><br />
- **Time alignment-enabled inference with NeMo models project:** You can generate transcripts with NeMo-based ASR models, such as Conformer-CTC, Conformer-Transducer, Hybrid FastFormer etc. and generate char and word-level time alignment information for the generated transcripts. This requires installing the NeMo framework. Please see the [**NeMo ASR Experiments**](https://github.com/abarcovschi/nemo_asr/blob/main/README.md#generating-time-alignments-for-transcriptions) project for more details.<br /><br />
**UPDATE:** By using [abarcovschi/nemo_asr/transcribe_speech_custom.py](https://github.com/abarcovschi/nemo_asr/blob/main/transcribe_speech_custom.py), multiple hypotheses can also be returned from beam search decoding, instead of the default best hypothesis offered by the original project's [NeMo/examples/asr/transcribe_speech.py](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/transcribe_speech.py).<br /><br />
- **Wav2Vec2 forced alignment:** You can run `wav2vec2_forced_alignment_libri.py` to generate time alignments for paired <audio, ground truth text> datasets whose transcripts are saved in LibriSpeech **OR** LibriTTS format **OR** use the output transcripts/hypotheses (as ground truth) located in a JSON file outputted by an ASR inference scripts such as `wav2vec2_infer_custom.py`, `whisper_time_alignment.py` or [abarcovschi/nemo_asr/transcribe_speech_custom.py](https://github.com/abarcovschi/nemo_asr/blob/main/transcribe_speech_custom.py) and force align the audio to these hypotheses.<br />
Run `python wav2vec2_forced_alignment_libri.py --help` for a description of the usage.<br />
**NOTE:** models from torchaudio **AND** custom wav2vec2 models checkpoints trained/finetuned using the fairseq framework are now supported.

### Output Formats

#### Inference

The format of JSON files containing word-level time-aligned transcripts, outputted by `wav2vec2_infer_custom.py` and `whisper_time_alignment.py`, is the following:<br /><br />
```json
{"wav_path": "/path/to/audio1.wav", "id": "unique/audio1/id", "pred_txt": "the predicted transcript sentence for audio one", "timestamps_word": [{"word": "the", "start_time": 0.1, "end_time": 0.2}, {"word": "predicted", "start_time": 0.3, "end_time": 0.4}, {"word": "transcript", "start_time": 0.5, "end_time": 0.6}, {"word": "sentence", "start_time": 0.7, "end_time": 0.8}, {"word": "for", "start_time": 0.9, "end_time": 1.0}, {"word": "audio", "start_time": 1.1, "end_time": 1.2}, {"word": "one", "start_time": 1.3, "end_time": 1.4}]}
{"wav_path": "/path/to/audio2.wav", "id": "unique/audio2/id", "pred_txt": "the predicted transcript sentence for audio two", "timestamps_word": [{"word": "the", "start_time": 0.1, "end_time": 0.2}, {"word": "predicted", "start_time": 0.3, "end_time": 0.4}, {"word": "transcript", "start_time": 0.5, "end_time": 0.6}, {"word": "sentence", "start_time": 0.7, "end_time": 0.8}, {"word": "for", "start_time": 0.9, "end_time": 1.0}, {"word": "audio", "start_time": 1.1, "end_time": 1.2}, {"word": "two", "start_time": 1.3, "end_time": 1.4}]}
```
etc.
<br /><br />
**NOTE1:** The 'timestamps_word' field is optionally outputted by decoders that support word-level timestamps creation **if** the `--time_aligns` flag is set using `wav2vec2_infer_custom.py`; and if the `--time_aligns` flag is set using `whisper_time_alignment.py`.<br /><br />
**NOTE2:** If `--num_hyps` is set to a value >1 when using `wav2vec2_infer_custom.py`; and if `--beam_size` is set to a value >1 **along** with `--num_hyps`<=`--beam_size` when using `whisper_time_alignment.py`, then **multiple** output files will be created, e.g. if `--num_hyps`=3:<br />
```
hypotheses1_of_3.json -> contains the best hypothesis per audio file in the input folder.
hypotheses2_of_3.json -> contains the second best hypothesis per audio file in the input folder.
hypotheses3_of_3.json -> contains the third best hypothesis per audio file in the input folder.
```
files are created, with each file containing a JSON row (dict containing 'wav_path', 'id', 'pred_txt', ['timestamps_word'] fields) per audio file in the input folder.<br /><br />
If `--num_hyps` is set to 1 when using `wav2vec2_infer_custom.py`, or if using a decoder that does not support returning multiple hypotheses; and if `--num_hyps` is set to 1 or if `--beam_size` is set to 1 when using `whisper_time_alignment.py`, then a **single** `best_hypotheses.json` file will be created, containing just the best hypothesis per audio file in the input folder.<br /><br />

For the output format produced using NeMo models using the [abarcovschi/nemo_asr/transcribe_speech_custom.py](https://github.com/abarcovschi/nemo_asr/blob/main/transcribe_speech_custom.py) script, please see the [**NeMo ASR Experiments**](https://github.com/abarcovschi/nemo_asr/blob/main/README.md#generating-time-alignments-for-transcriptions) project (output is similar, but has character-level timestamps option also).

#### Forced Alignment

The format of JSON files containing word-level force-aligned transcripts, outputted by `wav2vec2_forced_alignment_libri.py` is the following:<br /><br />
```json
{"wav_path": "/path/to/audio1.wav", "id": "unique/audio1/id", "ground_truth_txt": "the ground truth transcript for audio one", "alignments_word": [{"word": "the", "confidence": 0.88, "start_time": 0.1, "end_time": 0.2}, {"word": "ground", "confidence": 0.88, "start_time": 0.3, "end_time": 0.4}, {"word": "truth", "confidence": 0.88, "start_time": 0.5, "end_time": 0.6}, {"word": "transcript", "confidence": 0.88, "start_time": 0.7, "end_time": 0.8}, {"word": "for", "confidence": 0.88, "start_time": 0.9, "end_time": 1.0}, {"word": "audio", "confidence": 0.88, "start_time": 1.1, "end_time": 1.2}, {"word": "one", "confidence": 0.88, "start_time": 1.3, "end_time": 1.4}]}
{"wav_path": "/path/to/audio2.wav", "id": "unique/audio2/id", "ground_truth_txt": "the ground truth transcript for audio two", "alignments_word": [{"word": "the", "confidence": 0.88, "start_time": 0.1, "end_time": 0.2}, {"word": "ground", "confidence": 0.88, "start_time": 0.3, "end_time": 0.4}, {"word": "truth", "confidence": 0.88, "start_time": 0.5, "end_time": 0.6}, {"word": "transcript", "confidence": 0.88, "start_time": 0.7, "end_time": 0.8}, {"word": "for", "confidence": 0.88, "start_time": 0.9, "end_time": 1.0}, {"word": "audio", "confidence": 0.88, "start_time": 1.1, "end_time": 1.2}, {"word": "two", "confidence": 0.88, "start_time": 1.3, "end_time": 1.4}]}
```
etc.

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


