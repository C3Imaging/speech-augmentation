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
from dataclasses import dataclass

torch.random.manual_seed(0)
matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging():
    """Set up logging to both console and a logfile."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # track INFO logging events (default was WARNING)
    root_logger.handlers = []  # clear handlers
    root_logger.addHandler(logging.StreamHandler(sys.stdout))  # handler to log to console
    root_logger.addHandler(logging.FileHandler(os.path.join(arg_folder, 'wav2vec2_alignments_run.log'), 'w+'))  # handler to log to file also
    root_logger.handlers[0].setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))  # log level and message
    root_logger.handlers[1].setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


def plot_trellis_with_path(trellis, path):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for _, p in enumerate(path):
        trellis_with_path[p.time_index, p.token_index] = float("nan")
    plt.imshow(trellis_with_path[1:, 1:].T, origin="lower")


def plot_trellis_with_segments(path, trellis, segments, transcript):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))
    ax1.set_title("Path, label and probability for each label")
    ax1.imshow(trellis_with_path.T, origin="lower")
    ax1.set_xticks([])

    for i, seg in enumerate(segments):
        if seg.label != "|":
            ax1.annotate(seg.label, (seg.start + 0.7, i + 0.3), weight="bold")
            ax1.annotate(f"{seg.score:.2f}", (seg.start - 0.3, i + 4.3))

    ax2.set_title("Label probability with and without repetition")
    xs, hs, ws = [], [], []
    for seg in segments:
        if seg.label != "|":
            xs.append((seg.end + seg.start) / 2 + 0.4)
            hs.append(seg.score)
            ws.append(seg.end - seg.start)
            ax2.annotate(seg.label, (seg.start + 0.8, -0.07), weight="bold")
    ax2.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")

    xs, hs = [], []
    for p in path:
        label = transcript[p.token_index]
        if label != "|":
            xs.append(p.time_index + 1)
            hs.append(p.score)

    ax2.bar(xs, hs, width=0.5, alpha=0.5)
    ax2.axhline(0, color="black")
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(-0.1, 1.1)


def plot_alignments(trellis, segments, word_segments, waveform):
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))

    ax1.imshow(trellis_with_path[1:, 1:].T, origin="lower")
    ax1.set_xticks([])
    ax1.set_yticks([])

    for word in word_segments:
        ax1.axvline(word.start - 0.5)
        ax1.axvline(word.end - 0.5)

    for i, seg in enumerate(segments):
        if seg.label != "|":
            ax1.annotate(seg.label, (seg.start, i + 0.3))
            ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 4), fontsize=8)

    # The original waveform
    ratio = waveform.size(0) / (trellis.size(0) - 1)
    ax2.plot(waveform)
    for word in word_segments:
        x0 = ratio * word.start
        x1 = ratio * word.end
        ax2.axvspan(x0, x1, alpha=0.1, color="red")
        ax2.annotate(f"{word.score:.2f}", (x0, 0.8))

    for seg in segments:
        if seg.label != "|":
            ax2.annotate(seg.label, (seg.start * ratio, 0.9))
    xticks = ax2.get_xticks()
    plt.xticks(xticks, xticks / bundle.sample_rate)
    ax2.set_xlabel("time [second]")
    ax2.set_yticks([])
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_xlim(0, waveform.size(-1))


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


# Merge words
def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


def run_inference_batch(root_cur_out_dir, speech_files, transcripts):
    """Runs wav2vec2 batched inference on all audio files.

    The inference includes first generating the emission matrix, which estimates the probability of each label in the vocabulary for each time frame in the audio file.
     Then the trellis matrix is generated, which represents the probability of each label from the trancript file over all time steps/frames in the audio file.
     Then the most likely alignment is found by traversing the most likely path through the trellis matrix, which finds the most likely label at each timestep. Then the most
     likely characters are merged into words by collapsing repeating labels between '|' labels and the start and stop times of each word can thenbe saved.
     The transcript produced ideally should match the ground truth transcript from the transcripts list.
    
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

            # generate the label class probability of each audio frame using wav2vec2 for each label
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
            # plt.xlabel("Time")
            # plt.ylabel("Labels")
            # plt.show()
            
            # list of transcript characters in the order they occur in the transcript, where each element in list is the character's index in the vocabulary dictionary
            tokens = [dictionary[c] for c in transcript]
            # print(list(zip(transcript, tokens)))

            trellis = get_trellis(emission, tokens)
            # plt.imshow(trellis[1:, 1:].T, origin="lower")
            # plt.annotate("- Inf", (trellis.size(1) / 5, trellis.size(1) / 1.5))
            # plt.colorbar()
            # plt.show()

            # the trellis matrix is used for path-finding, but for the final probability of each segment, we take the frame-wise probability from emission matrix.
            path = backtrack(trellis, emission, tokens)
            # for p in path:
            #     print(p)

            # plot_trellis_with_path(trellis, path)
            # plt.title("The path found by backtracking")

            segments = merge_repeats(path, transcript)
            # for seg in segments:
            #     print(seg)

            word_segments = merge_words(segments)

            # for debugging purposes, --save_figs flag
            if save_figs:
                plot_trellis_with_segments(path, trellis, segments, transcript)
                plt.tight_layout()
                plt.savefig(os.path.join(cur_out_dir, 'trellis_with_frames.png'))

                plot_alignments(
                    trellis,
                    segments,
                    word_segments,
                    waveform[0],
                )
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
                    if save_audio:
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

    The model, which was trained for ASR, first makes character predictions on an audio file, then predicts the time alignment of those characters.
    The script then joins the characters into words and postprocesses the wav2vec2 predictions to get the start and stop times of each word in an audio file.
    The output will then be a text file in CSV format for each audio file that saves the time alignments of each word.

    The folder specified at the command line must be in Librispeech format, where it is either a leaf folder that contains at least one .flac audio file and only one .txt transcript file,
     or a folder containing a directory tree where there are many leaf folders.
    The script can be run on either one leaf folder (specified by the 'leaf' --mode at the command line) or on a dirtectory tree, where there are multiple leaf folders,
     effectively running the script over an entire dataset (specified by the 'root' --mode at the command line).
    """
    # if mode=leaf run the script only on the audio files in a single folder specified
    # if mode=root, run the script on all subfolders, essentially over the entire dataset

    for dirpath, _, filenames in os.walk(arg_folder, topdown=asleaf): # if topdown=True, read contents of folder before subfolders, otherwise the reverse logic applies
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
    "setup and use wav2vec2 model"
    # setup inference model variables
    global bundle, model, labels, dictionary
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H # wav2vec2 model trained for ASR, sample rate = 16kHz
    model = bundle.get_model().to(device) # wav2vec2 model on GPU
    labels = bundle.get_labels() # vocab of chars known to wav2vec2
    dictionary = {c: i for i, c in enumerate(labels)}
    
    run_inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run demo inference of wav2vec2 forced alignment on Librispeech dataset folder format.")
    parser.add_argument("folder", type=str, nargs='?', default=os.getcwd(),
                        help="Path to a folder in Librispeech, can be a root folder containing other folders or a leaf folder containing audio and transcript files. Defaults to CWD if not provided.")
    parser.add_argument("--mode", type=str, choices={'leaf', 'root'}, default="root",
                        help="Specifies how the folder will be processed.\nIf 'leaf': only the folder will be searched for audio files (single folder inference),\nIf 'root': subdirs are searched (full dataset inference).\nDefaults to 'root' if unspecified.")
    parser.add_argument("--save_figs", default=False, action='store_true',
                        help="Flag used to specify whether graphs of alignments are saved for each audio file. Defaults to False if flag is not provided.")
    parser.add_argument("--save_audio", default=False, action='store_true',
                        help="Flag used to specify whether detected words are saved as audio snippets. Defaults to False if flag is not provided.")
    # parse command line arguments
    args = parser.parse_args()
    # setup folder structure variables
    global arg_folder, out_dir
    arg_folder = args.folder # the folder specified as a command line argument
    out_dir = "wav2vec2_alignments" # the output folder to be created in folders where there are audio files and a transcript file
    # setup logging
    setup_logging()
    # setup mode variables
    mode = args.mode
    global asleaf
    asleaf = True if mode == 'leaf' else False
    # setup debugging variables
    global save_figs, save_audio
    save_figs = args.save_figs # boolean
    save_audio = args.save_audio # boolean

    # start timing how long it takes to run script
    tic = time.perf_counter()

    logging.info("started script.")
    main()
    logging.info("ended script.")

    toc = time.perf_counter()
    logging.info(f"Finished processing in {time.strftime('%H:%M:%Ss', time.gmtime(toc - tic))}")