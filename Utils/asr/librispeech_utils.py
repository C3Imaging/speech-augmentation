import os
import json


def get_speech_data_lists_from_json(filepath):
    """Returns a list of tuples, where tuples contain corresponding wavpath and transcript string from a 'hypotheses.json' inference transcript file from the output of running 'wav2vec2_infer_custom.py', 'whisper_time_alignment.py' or 'https://github.com/abarcovschi/nemo_asr/blob/main/transcribe_speech_custom.py'.

    Args:
      filepath (str):
        The path to a hypotheses.json inference transcript file.

    Returns:
      speech_files (str, list):
        A sorted list of speech file paths.
      transcripts (str, list):
        A list of transcript strings corresponing to each speech file.
        The format of the strings is the one used by wav2vec2 forced alignment tutorial at https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html

    """
    speech_files = []
    transcripts = []
    # read transcript file line by line
    with open(filepath, 'r', encoding='utf-8') as fr:
        for line in fr:
            item = json.loads(line)

            wav_path = item['wav_path']

            # join words using '|' symbol as wav2vec2 uses this symbol as the word boundary
            words = '|'.join(item['pred_txt'].split(' ')).upper()
            speech_files.append(wav_path)
            transcripts.append(words)

    return speech_files, transcripts


def get_transcripts(filepath):
    """Returns a list of transcript strings from a Librispeech transcript file, which contains the transcripts for all speech files in a leaf folder.
    
    Args:
      filename (str):
        The path to a Librispeech transcript file.
    Returns:
      transcripts (str, list):
        A list of transcript strings corresponing to each speech file.
        The format of the strings is the one used by wav2vec2 forced alignment tutorial at https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
    """
    transcripts = []
    # read transcript file line by line
    with open(filepath) as f:
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
    """Gets the speech audio files paths and the transcripts from a single leaf folder in Librispeech format.

    Args:
      dirpath (str):
        The path to the directory containing the audio files and transcript file.
      filenames (list of str elements):
        the list of all files found by os.walk in this directory.

    Returns:
      speech_files (str, list):
        A sorted list of speech file paths found in this directory.
      transcripts (str, list):
        A list of transcript strings corresponding to each speech file.
    """ 
    speech_files = []
    transcript, transcripts = None, None
    # idx, transcript_filename = [(idx, os.path.join(path, filename)) for idx, filename in enumerate(filenames) if filename.endswith('.txt')][0]
    # del filenames[idx] # in place removal of transcript file by its index, creates speech filenames list
    # loop through all files found
    for filename in filenames:
        if filename.endswith('.flac') or filename.endswith('.wav'):
            speech_files.append(os.path.join(dirpath, filename))
        elif filename.endswith('.txt'):
            transcript = os.path.join(dirpath, filename)

    # check if it is a leaf folder
    if transcript is not None:
        transcripts = get_transcripts(transcript)
        speech_files.sort()

    return speech_files, transcripts