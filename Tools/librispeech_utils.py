import os


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


def get_transcripts_from_w2v2_inference(filename):
    """Returns a list of tuples, where tuples contain corresponding wavpath and transcript string from a 'hypothesis.txt' transcript file from the output of running 'wav2vec2_infer_custom.py'.

    The format of the processed transcript strings is the one used by wav2vec2 forced alignment tutorial at https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html

    Args:
      filename (str):
        The path to a hypothesis.txt transcript file outputted after running 'wav2vec2_infer_custom.py'.

    Returns:
      speech_files (str, list):
        A sorted list of speech file paths.
      transcripts (str, list):
        A list of transcript strings corresponing to each speech file.
    """
    speech_files = []
    transcripts = []
    # read transcript file line by line
    with open(filename) as f:
        for line in f:
            words = line.split(" ")
            # remove id
            del words[1]

            wav_path = words[0].replace("(",'').replace(")",'')
            # remove wavpath from the transcript
            del words[0]

            # remove \n from the last word
            words[-1] = words[-1].replace("\n",'')



            # join words using '|' symbol as wav2vec2 uses this symbol as the word boundary
            words = '|'.join(words).upper()
            speech_files.append(wav_path)
            transcripts.append(words)

    return speech_files, transcripts


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
        A list of transcript strings corresponing to each speech file.
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


if __name__ == "__main__":
    get_transcripts_from_w2v2_inference("/workspace/datasets/LibriSpeech_test/w2v2_infer_out/hypothesis.txt")