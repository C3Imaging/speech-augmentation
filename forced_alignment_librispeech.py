# align transcript to speech with CTC segmentation algorithm
# Wav2Vec2 model is used for acoustic feature extraction
# based on the tutorial: https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html

import os
import sys
import time
import torch
import logging
import argparse
import torchaudio
import matplotlib
import matplotlib.pyplot as plt
from Tools import utils
from Tools import forced_alignment_utils


def run_inference_batch(root_cur_out_dir, speech_files, transcripts):
    """Runs wav2vec2 batched inference on all audio files read from a leaf folder.

    The inference includes first generating the emission matrix by the wav2vec2 model, which contains the log probabilities of each label in the vocabulary for each time frame in the audio file.
     Then the trellis matrix is generated, which encodes the probabilities of each character from the transcript file over all time steps/frames in the audio file.
     Then the most likely alignment is found by traversing the most likely path through the trellis matrix, which finds the most likely character at each timestep by selecting the ground truth
      transcript character with the highest probability at each timestep from the trellis matrix. 
     Then repeating labels for the same character are first merged into one ground truth transcript character and then those are merged into words and the start and stop times of each word can then be saved.

     NOTE: We force the alignment between the ground truth transcript and timesteps because we only concentrate on using the probabilities of the ground truth transcript characters at each timestep, disregarding
     all other possible alignments which may have tokens with higher probabilities in the emission matrix at a particular timestep, thus this is called wav2vec2 forced alignment.
           The model's output is not processed by any decoding algorithms to find a transcript (e.g. beam search, greedy search) because we have a ground truth transcript.
    
    The alignments of each word in each speech file is saved into the output directory. Optionally, plots of the alignment between words predicted by wav2vec2 and the audio file
     can be saved as images into the output directory if the --save_figs flag is set and each word segment extracted from the audio file can be saved as a separate audio file snippet
     into the output directory if the --save_audio flag is set.

    Args:
      root_cur_out_dir (str):
        The name of the output directory into which the alignments will be saved.
      speech_files (str, list):
        A sorted list of speech file paths found in this directory.
      transcripts (str, list):
        A list of transcript strings in wav2vec2 format corresponing to each speech file.
    """
    with torch.inference_mode():
        # loop through audio files
        for speech_filename, transcript in zip(speech_files, transcripts):
            # create separate subdir for experiment outputs for this speech file
            # get speech segment index from the filename
            speech_idx = speech_filename.split('.flac')[0].split('-')[-1]
            # create output subfolder for an audio file
            cur_out_dir = os.path.join(root_cur_out_dir, speech_idx)
            if not os.path.exists(cur_out_dir): os.makedirs(cur_out_dir, exist_ok=True)

            # generate the label class probability of each audio frame using wav2vec2 for each label (outputs are actually in logits, not probabilities)
            waveform, _ = torchaudio.load(speech_filename)
            emissions, _ = model(waveform.to(device))
            emissions = torch.log_softmax(emissions, dim=-1) # probability in log domain to avoid numerical instability
            # probability of each vocabulary label at each time step
            # for silences, wav2vec2 predicts the '|' label with very high probability, which is the word boundary label
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

            # for debugging purposes, --save_figs flag
            if args.save_figs:
                forced_alignment_utils.plot_trellis_with_path(trellis, path)
                plt.savefig(os.path.join(cur_out_dir, 'trellis_with_path.png'))

                forced_alignment_utils.plot_trellis_with_segments(path, trellis, segments, transcript)
                plt.savefig(os.path.join(cur_out_dir, 'trellis_with_frames.png'))

                forced_alignment_utils.plot_alignments(bundle, trellis, segments, word_segments, waveform[0])
                plt.savefig(os.path.join(cur_out_dir, 'trellis_with_waveform.png'))
            
            with open(os.path.join(cur_out_dir, 'alignments.txt'), 'w') as f:
                f.write("confidence_score,word_label,start_time,stop_time\n") # time is in seconds
                i = 0
                # for each word detected, save to file the {confidence score, label, start time, stop time} as a CSV line
                for word in word_segments:
                    ratio = waveform.size(1) / (trellis.size(0) - 1)
                    x0 = int(ratio * word.start)
                    x1 = int(ratio * word.end)
                    f.write(f"{word.score:.2f},{word.label},{x0 / bundle.sample_rate:.3f},{x1 / bundle.sample_rate:.3f}\n")
                    # for debugging purposes, --save_audio flag
                    if args.save_audio:
                        # save snippet of audio where only the word is present
                        segment = waveform[:, x0:x1]
                        torchaudio.save(os.path.join(cur_out_dir, f"word{i}_{word.label}.wav"), segment, bundle.sample_rate)
                        i+=1


def get_transcripts(filename):
    """Returns a list of transcript strings from a Librispeech transcript file, which contains the transcripts for all speech files in a leaf folder.

    The format of the processed transcript strings is the one used by wav2vec2 forced alignment tutorial at https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html

    Args:
      filename (str):
        The path to a Librispeech transcript file.
    """
    transcripts = []
    # read transcript file line by line
    with open(filename) as f:
        for line in f:
            words = line.split(" ")
            # remove non-word first element
            del words[0]
            # remove \n from the last word
            words[-1] = words[-1].replace("\n",'')
            #pipe_delimited_words = []
            # for w in words:
            #     pipe_delimited_words.append(w)
            #     pipe_delimited_words.append('|')
            # del pipe_delimited_words[-1]

            # join words using '|' symbol as wav2vec2 uses this symbol as the word boundary
            words = '|'.join(words)
            transcripts.append(words)

    return transcripts


def get_speech_data_lists(dirpath, filenames):
    """Gets the speech audio files paths and the transcripts from a single leaf folder.

    Args:
      dirpath (str):
        The path to the directory containing the audio files and transcript file.
      filenames (list of str elements):
        the list of all files found by os.walk in this directory.

    Returns:
      speech_files (str, list):
        A sorted list of speech file paths found in this directory.
      transcripts (str, list):
        A list of transcript strings corresponing to each speech file.
    """
    speech_files = []
    transcript, transcripts = None, None
    # idx, transcript_filename = [(idx, os.path.join(path, filename)) for idx, filename in enumerate(filenames) if filename.endswith('.txt')][0]
    # del filenames[idx] # in place removal of transcript file by its index, creates speech filenames list
    # loop through all files found
    for filename in filenames:
        if filename.endswith('.flac'):
            speech_files.append(os.path.join(dirpath, filename))
        elif filename.endswith('.txt'):
            transcript = os.path.join(dirpath, filename)

    # check if it is a leaf folder
    if transcript is not None:
        transcripts = get_transcripts(transcript)
        speech_files.sort()

    return speech_files, transcripts


def run_inference():
    """Runs wav2vec2 model on a folder specified as the positional argument of this script.

    The model, which was trained for ASR, first makes character predictions on an audio file, then the time alignment between the ground truth transcript characters and the audio is created.
    The code processes the wav2vec2 predictions to get the start and stop times of each ground truth transcript character in an audio file then joins them into words.
    The output will then be a text file in CSV format for each audio file that saves the time alignments of each word.

    The folder specified at the command line must be in Librispeech format, where it is either a 'leaf' folder that contains at least one .flac audio file and only one .txt transcript file,
     or a 'root' folder containing a directory tree where there are many leaf subfolders.
    The script can be run on either one leaf folder (specify the 'leaf' --mode at the command line) or on a root folder, where there are multiple leaf folders,
     effectively running the script over the entire dataset (specified the 'root' --mode at the command line).
    """
    # if mode=leaf run the script only on the audio files in a single folder specified
    # if mode=root, run the script on all subfolders, essentially over the entire dataset

    for dirpath, _, filenames in os.walk(args.folder, topdown=asleaf): # if topdown=True, read contents of folder before subfolders, otherwise the reverse logic applies
        # if this script was run previously, an output folder will be present in the folder where the audio files we want to process are, skip it and its subfolders
        if out_dir not in dirpath:
            # get list of speech files and corresponding transcripts from a single folder
            speech_files, transcripts = get_speech_data_lists(dirpath, filenames)
            # process only those folders that contain a transcripts text file
            if transcripts is not None:
                logging.info(f"starting to process folder {dirpath}")
                # create root output dir (specified by global out_dir)
                cur_out_dir = os.path.join(dirpath, out_dir)
                if not os.path.exists(cur_out_dir): os.makedirs(cur_out_dir, exist_ok=True)
                # run wav2vec2
                run_inference_batch(cur_out_dir, speech_files, transcripts)
                logging.info(f"finished processing folder {dirpath}")
            if asleaf:
                break # to prevent reading subfolders


def main():
    "Setup and use wav2vec2 model for time alignment between ground truth transcript and audio file from Librispeech dataset."
    # setup inference model variables
    global bundle, model, labels, dictionary
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H # wav2vec2 model trained for ASR, sample rate = 16kHz
    model = bundle.get_model().to(device) # wav2vec2 model on GPU
    labels = bundle.get_labels() # vocab of chars known to wav2vec2
    dictionary = {c: i for i, c in enumerate(labels)}
    
    run_inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ASR inference using wav2vec2 ASR model and perform forced alignment on folder(s) in the Librispeech dataset. NOTE: this script can only use wav2vec2 ASR models from torchaudio library.")
    parser.add_argument("folder", type=str, nargs='?', default=os.getcwd(),
                        help="Path to a folder in Librispeech, can be a root folder containing other folders or a leaf folder containing audio and transcript files. Defaults to CWD if not provided.")
    parser.add_argument("--mode", type=str, choices={'leaf', 'root'}, default="root",
                        help="Specifies how the folder will be processed.\nIf 'leaf': only the folder will be searched for audio files (single folder inference),\nIf 'root': subdirs are searched (full dataset inference).\nDefaults to 'root' if unspecified.")
    parser.add_argument("--save_figs", default=False, action='store_true',
                        help="Flag used to specify whether graphs of alignments are saved for each audio file. Defaults to False if flag is not provided.")
    parser.add_argument("--save_audio", default=False, action='store_true',
                        help="Flag used to specify whether detected words are saved as audio snippets. Defaults to False if flag is not provided.")
    # parse command line arguments
    global args
    args = parser.parse_args()
    
    # setup folder structure variables
    global out_dir
    out_dir = "wav2vec2_alignments" # the output folder to be created in folders where there are audio files and a transcript file

    # setup logging to both console and logfile
    utils.setup_logging(args.folder, 'wav2vec2_forced_alignment_librispeech.log', console=True)

    # setup directory traversal mode variables
    mode = args.mode
    global asleaf
    asleaf = True if mode == 'leaf' else False

    #setup CUDA and Matplotlib configs
    torch.random.manual_seed(0)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

    # start timing how long it takes to run script
    tic = time.perf_counter()

    # log the command that started the script
    logging.info(f"Started script via: {' '.join(sys.argv)}")
    main()

    toc = time.perf_counter()
    logging.info(f"Finished processing in {time.strftime('%H:%M:%Ss', time.gmtime(toc - tic))}")