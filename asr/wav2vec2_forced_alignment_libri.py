"""
Force align transcript returned by a Wav2Vec2 ASR acoustic model to speech at the word level using the CTC segmentation algorithm.
Based on the tutorial: https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html

For a more detailed explanation on how forced-alignment works see: https://nvidia.github.io/NeMo/blogs/2023/2023-08-forced-alignment/
"""

import os
import sys
import json
import torch
import logging
import argparse
import torchaudio
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Utils import utils
from Utils.asr.decoding_utils_w2v2 import Wav2Vec2_Decoder_Factory
from Utils.asr import forced_alignment_utils, librispeech_utils, libritts_utils


def run_inference(args, model, labels, dictionary, sr, speech_files, transcripts):
    """Runs inference on all speech files and force aligns the corresponding transcripts.

    The model, which was trained for ASR, first makes character predictions on an audio file, then the time alignment between the ground truth transcript characters and the audio is created.
    The code processes the wav2vec2 predictions to get the start and stop times of each ground truth transcript character in an audio file then joins them into words.
    The output will then be a JSON file with a line for each audio file, where the time alignments of each word are saved.

    The inference includes first generating the emission matrix by the wav2vec2 model, which contains the log probabilities of each label in the vocabulary for each time frame in the audio file.
     Then the trellis matrix is generated, which encodes the probabilities of each character from the transcript file over all time steps/frames in the audio file.
     Then the most likely alignment is found by traversing the most likely path through the trellis matrix, which finds the most likely character at each timestep by selecting the ground truth
      transcript character with the highest probability at each timestep from the trellis matrix. 
     Then repeating labels for the same character are first merged into one ground truth transcript character and then those are merged into words and the start and stop times of each word can then be saved.

     NOTE: We force the alignment between the ground truth transcript and timesteps because we only concentrate on using the probabilities of the ground truth transcript characters at each timestep, disregarding
     all other possible alignments which may have tokens with higher probabilities in the emission matrix at a particular timestep, thus this is called forced alignment.
        The model's emission output is not processed by any decoding algorithms to find a transcript (e.g. beam search, greedy search) because we have a ground truth transcript.
    
    The alignments of each word in each speech file is saved into the output directory. Optionally, plots of the alignment between words predicted by wav2vec2 and the audio file
     can be saved as images into the output directory if the --save_figs flag is set and each word segment extracted from the audio file can be saved as a separate audio file snippet
     into the output directory if the --save_audio flag is set.

    Args:
      speech_files [List[str]]:
        A sorted list of speech file paths.
      transcripts [List[str]]:
        A list of transcript strings in wav2vec2 format, corresponing to each speech file.
    """
    with torch.inference_mode():
        # loop through audio files
        for speech_filename, transcript in tqdm(zip(speech_files, transcripts), total=len(speech_files), unit=" <audio, transcript> pair", desc="Forced aligning <audio, transcript> pairs sequentially, so far"):
            logging.info(f"forced aligning {speech_filename}")

            if args.model_path:
                # using custom w2v2 model from a checkpoint (fairseq framework).
                emissions = model.forward([speech_filename], device)
            else:
                # using torchaudio w2v2 model.
                # generate the label class probability of each audio frame using wav2vec2 for each label (outputs are actually in logits, not probabilities).
                waveform, _ = torchaudio.load(speech_filename)
                emissions, _ = model(waveform.to(device))
                emissions = torch.log_softmax(emissions, dim=-1) # probability in log domain to avoid numerical instability.
            # probability of each vocabulary label at each time step
            # for silences, wav2vec2 predicts the '|' label with very high probability, which is the word boundary label.
            if emissions[0].size(0) == 1:
                # custom fairseq checkpoint returns a list of 2 elements, where [0] is the batched emission matrix.
                emission = emissions[0].cpu().detach()[0]
            else:
                # torchaudio returns just the batched emission matrix.
                emission = emissions[0].cpu().detach()
            # print(labels)
            # plt.imshow(emission.T)
            # plt.colorbar()
            # plt.title("Frame-wise class probability")
            # plt.xlabel("Time (frames)")
            # plt.ylabel("Labels from wav2vec2 vocabulary")
            # plt.show()
            
            # list of transcript characters in the order they occur in the transcript, where each element in list is the character's index in the vocabulary dictionary
            tokens = [dictionary[c] for c in transcript]
            # print(list(zip(transcript, tokens)))

            trellis = forced_alignment_utils.get_trellis(emission, tokens)
            # plt.imshow(trellis[1:, 1:].T, origin="lower")
            # plt.annotate("- Inf", (trellis.size(1) / 5, trellis.size(1) / 1.5))
            # plt.colorbar()
            # plt.title("trellis matrix, colour represents confidence score")
            # plt.xlabel("frame")
            # plt.ylabel("char index in ground truth transcript")
            # plt.show()

            # the trellis matrix is used for path-finding, but for the final probability of each segment, we take the frame-wise probability from emission matrix.
            path = forced_alignment_utils.backtrack(trellis, emission, tokens)
            # for p in path:
            #     print(p)

            segments = forced_alignment_utils.merge_repeats(path, transcript)
            # for seg in segments:
            #     print(seg)

            word_segments = forced_alignment_utils.merge_words(segments)

            waveform, _ = torchaudio.load(speech_filename)

            # create unique id of this audio by including leaf folder of the audio filepath in the id.
            # this will be used as the subfolder in args.out_dir into which to save the figures for this audio file.
            temp = speech_filename.split('/')[-2:] # [0] = subfolder, [1] = ____.wav/____.flac
            temp[-1] = temp[-1].split('.wav')[0] if '.wav' in temp else temp[-1].split('.flac')[0] # remove '.wav' or '.flac'
            id = '/'.join(temp)

            with open(os.path.join(args.out_dir, 'forced_alignments.json'), 'a') as f:
                item = dict()
                item['wav_path'] = speech_filename
                item['id'] = id
                item['ground_truth_txt'] = ' '.join(transcript.split('|')).strip().lower()
                vals = list()
                for word in word_segments:
                    word_dict = dict()
                    ratio = waveform.size(1) / (trellis.size(0) - 1)
                    x0 = int(ratio * word.start)
                    x1 = int(ratio * word.end)
                    start_time = x0 / sr
                    end_time = x1 / sr
                    word_dict['word'] = word.label.lower()
                    word_dict['confidence'] = word.score
                    word_dict['start_time'] = start_time
                    word_dict['end_time'] = end_time
                    vals.append(word_dict)
                item['alignments_word'] = vals
                f.write(json.dumps(item) + "\n")

            # for debugging purposes
            if args.save_figs:
                # cannot add a '/' symbol to the subfolder name, so replace it with 'SLASH'.
                id = id.replace("/", "SLASH")
                
                # create output subfolder for an audio file.
                cur_out_dir = os.path.join(args.out_dir, id)
                if not os.path.exists(cur_out_dir): os.makedirs(cur_out_dir, exist_ok=True)

                forced_alignment_utils.plot_trellis_with_path(trellis, path)
                plt.savefig(os.path.join(cur_out_dir, 'trellis_with_path.png'))

                forced_alignment_utils.plot_trellis_with_segments(path, trellis, segments, transcript)
                plt.savefig(os.path.join(cur_out_dir, 'trellis_with_frames.png'))

                forced_alignment_utils.plot_alignments(sr, trellis, segments, word_segments, waveform[0])
                plt.savefig(os.path.join(cur_out_dir, 'trellis_with_waveform.png'))

            logging.info(f"finished forced aligning {speech_filename}")


def get_audio_txt_pairs(args):
    """If using a folder input, the folder specified at the command line must be in Librispeech format or LibriTTS format, where it is either a 'leaf' folder that contains at least one .flac or .wav audio file and only one .txt transcript file,
     or a 'root' folder containing a directory tree where there are many leaf subfolders.
     The script can be run on either one leaf folder (specify the 'leaf' --mode at the command line) or on a root folder, where there are multiple leaf folders,
      effectively running the script over the entire dataset (specified the 'root' --mode at the command line).

    If using a JSON file input, the --mode and folder input is ignored.
    """
    all_speech_files, all_transcripts = [], []
    if args.input_from_json:
        # get list of speech files and corresponding transcripts from a hypotheses.json file.
        all_speech_files, all_transcripts = librispeech_utils.get_speech_data_lists_from_json(args.input_from_json)
    else:
        # 'folder' arg used.
        # if mode=leaf run the script only on the audio files in a single folder specified
        # if mode=root, run the script on all subfolders, essentially over the entire dataset
        for dirpath, _, filenames in os.walk(args.folder, topdown=args.asleaf): # if topdown=True, read contents of folder before subfolders, otherwise the reverse logic applies
            # process only folders that contain audio files or a hypothesis.txt file from custom wav2vec2 inference.      
            test = ' '.join(filenames)
            if ".wav" in test or ".flac" in test:
                # get list of speech files and corresponding transcripts from a single folder.
                if args.libritts:
                    speech_files, transcripts = libritts_utils.get_speech_data_lists(dirpath, filenames)
                else:
                    speech_files, transcripts = librispeech_utils.get_speech_data_lists(dirpath, filenames)

                all_speech_files.append(speech_files)
                all_transcripts.append(transcripts)

                if args.asleaf:
                    break # to prevent reading subfolders
        
        # flatten list of lists into a list.
        all_speech_files = [speechfile for minilist in all_speech_files for speechfile in minilist]
        all_transcripts = [transcript for minilist in all_transcripts for transcript in minilist]

    return all_speech_files, all_transcripts


def main(args):
    "Setup and use wav2vec2 model for time alignment between ground truth transcript and audio file from Librispeech or LibriTTS-formatted dataset or from JSON ASR inference output file."

    # setup model.

    # create model + decoder pair (change manually).
    if args.model_path:
        asr = Wav2Vec2_Decoder_Factory.get_cfg_beamsearchtransformerlm(model_filepath=args.model_path, vocab_path=args.vocab_path)
        # use only the ASR acoustic model.
        model = asr.model
        labels = list()
        with open(args.vocab_path, 'r') as file: lines = file.readlines()
        for line in lines:
            line = line.split(' ')[0]
            labels.append(line)
        dictionary = {c: i for i, c in enumerate(labels)}
        sr = asr.model.sample_rate
    else:
        # default to torchaudio wav2vec2 model if not using a custom .pt checkpoint.
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H # wav2vec2 model trained for ASR, sample rate = 16kHz
        # use only the ASR acoustic model.
        model = bundle.get_model().to(device) # wav2vec2 model on GPU
        labels = bundle.get_labels() # vocab of chars known to wav2vec2
        dictionary = {c: i for i, c in enumerate(labels)}
        sr = bundle.sample_rate

    # get lists of audio filepaths and corresponding ground truth transcripts strings to align.
    speech_files, transcripts = get_audio_txt_pairs(args)

    # run ASR inference and forced alignment if there is parallel <audio,txt> data available to align.
    if speech_files and transcripts:
        run_inference(args, model, labels, dictionary, sr, speech_files, transcripts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform forced alignment between ground-truth transcripts and the ASR acoustic model output using the CTC segmentation algorithm. <audio, ground truth transcript> data is read either from folder(s) structured in Librispeech or LibriTTS format or using <audio, transcript> data from a previously created hypotheses.json file after running an ASR inference script. NOTE1: Script is updated with LibriTTS format support, but must provide a '--libritts' flag. NOTE2: Script is updated to use both wav2vec2 ASR models from torchaudio library and custom fairseq/flashlight models.")
    parser.add_argument("folder", type=str, nargs='?', default=os.getcwd(),
                        help="From where <audio, ground truth transcripts> data is taken: path to a folder structured in Librispeech (only one transcripts.txt file in each audio subfolder) or LibriTTS (an individual transcript.txt file for each audio file) format. Can be a root folder containing other subfolders, such as speaker subfolders or recording session subfolders (use --mode='root' in this case), or a leaf folder containing audio and transcript file(s) (use --mode='leaf' in this case). Defaults to CWD if not provided.")
    parser.add_argument("--input_from_json", type=str, default='',
                        help="From where <audio, ground truth transcripts> data is taken: input data is now taken from the full path to a hypotheses.json file, which was the result of running inference using 'wav2vec2_infer_custom.py', 'whisper_time_alignment.py' or 'https://github.com/abarcovschi/nemo_asr/blob/main/transcribe_speech_custom.py', i.e. script treats output hypotheses in the JSON file as the ground truth transcripts. If used, the 'folder' path, '--mode' and '--libritts' flags are all ignored. Defaults to '' (i.e. 'folder' is used as the input data source).")
    parser.add_argument("--out_dir", type=str, required=True,
                    help="Path to a new output folder to create, where results will be saved.")
    parser.add_argument("--mode", type=str, choices={'leaf', 'root'}, default="root",
                        help="Specifies how the folder will be processed.\nIf 'leaf': only the folder will be searched for audio files (single folder inference),\nIf 'root': subdirs are searched (full dataset inference).\nDefaults to 'root' if unspecified. Flag is ignored if using '--input_from_json'")
    parser.add_argument("--libritts", default=False, action='store_true',
                        help="Flag used to specify whether the dataset is in LibriTTS format. Defaults to False (i.e. Librispeech) if flag is not provided. Flag is ignored if using '--input_from_json'.")
    parser.add_argument("--model_path", type=str, default='',
                        help="Path of a finetuned wav2vec2 model's .pt file. If unspecified, by default the script will use WAV2VEC2_ASR_LARGE_LV60K_960H torchaudio w2v2 model.")
    parser.add_argument("--vocab_path", type=str, default='',
                        help="Path of the finetuned wav2vec2 model's vocabulary text file (usually saved as dict.ltr.txt) that was used during wav2vec2 finetuning. Must be provided if '--model_path' arg is used.")
    parser.add_argument("--save_figs", default=False, action='store_true',
                        help="Flag used to specify whether graphs of alignments are saved for each audio file. Defaults to False if flag is not provided.")

    
    # parse command line arguments
    args = parser.parse_args()

    # check arg vals.
    if args.model_path and not args.vocab_path:
        raise ValueError("'--vocab_path' must be provided !!!")

    # setup logging to both console and logfile
    utils.setup_logging(args.out_dir, 'wav2vec2_forced_alignment_libri.log', console=True, filemode='w')

    # setup directory traversal mode variables
    mode = args.mode
    d = vars(args)
    d['asleaf'] = True if mode == 'leaf' else False

    #setup CUDA and Matplotlib configs
    torch.random.manual_seed(0)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

    p = utils.Profiler()
    p.start()

    main(args)

    p.stop()