import os


def convert_transcripts(transcripts_files):
    """Reads transcript files and converts each transcript from LibriTTS format to Librispeech format and returns the list."""
    transcripts = []
    for ff in transcripts_files:
        with open(ff) as f:
            tr = f.readline().strip().upper()
            # suppress_tokens_list = ['"', '?', '-', '.', ',', '.']
            tr2 = tr.translate({ord(i): None for i in '?!"-.,;:'}).replace(' ', '|')
            transcripts.append(tr2)
    
    return transcripts


def get_speech_data_lists(dirpath, filenames):
    """Gets the speech audio files paths and the transcripts from a single leaf folder in LibriTTS format.

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
    transcripts_files = []
    transcripts = None
    # idx, transcript_filename = [(idx, os.path.join(path, filename)) for idx, filename in enumerate(filenames) if filename.endswith('.txt')][0]
    # del filenames[idx] # in place removal of transcript file by its index, creates speech filenames list
    # loop through all files found
    for filename in filenames:
        if filename.endswith('.flac') or filename.endswith('.wav'):
            speech_files.append(os.path.join(dirpath, filename))
        elif filename.endswith('.original.txt'):
            transcripts_files.append(os.path.join(dirpath, filename))

    # check if it is a leaf folder
    if len(transcripts_files):
        transcripts_files.sort()
        speech_files.sort()
        transcripts = convert_transcripts(transcripts_files)
        
    return speech_files, transcripts
