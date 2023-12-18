import os
import re
from num2words import num2words


def try_convert_to_num(s):
  """Converts a string that has decimal numbers to the English spelling"""
  
  def _postproc(s):
    return s.replace('-', ' ').split(' ')
  
  # convert numbers with ordinal suffix characters
  n = None
  if s[-2:] in ['ST', 'ND', 'RD', 'TH']:
    try:
        n = int(s[:-2])
    except ValueError:
        pass
    if n is not None:
      return _postproc(num2words(n, ordinal=True).upper())
  # convert any number without suffixes
  try:
    return _postproc(num2words(s).upper())
  # parameter is not a number
  except Exception:
    return s


def convert_transcripts(transcripts_files):
    """Reads transcript files and converts each transcript from LibriTTS format to Librispeech format and returns the list."""
    
    def _flatten(tr_list):
      tr_list2 = []
      for element in tr_list:
        if isinstance(element, list):
          for word in element:
            tr_list2.append(word)
        else:
          tr_list2.append(element)
      return tr_list2
    
    transcripts = []
    for ff in transcripts_files:
      with open(ff) as f:
        # # long version
        # tr = f.readline().strip().upper()
        # # suppress tokens and replace space with pipe
        # tr = tr.translate({ord(i): None for i in '?!".,;:[]()—{}/\\'})
        # # replace dashes with spaces and accented Es with E
        # tr = tr.translate({ord(i): ' ' for i in '—-'})
        # tr = tr.translate({ord(i): 'E' for i in 'ÊÉ'})
        # # replace Æ with AE, Œ with OE
        # tr = tr.replace('Æ', 'AE')
        # tr = tr.replace('Œ', 'OE')
        # # compress multiple spaces into one
        # _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
        # tr = _RE_COMBINE_WHITESPACE.sub(" ", tr).strip()
        # # convert numbers to words
        # tr_list = tr.split(' ')
        # tr_list2 = list(map(try_convert_to_num, tr_list))
        # # flatten any sublists returned from number to word conversion
        # tr_list3 = _flatten(tr_list2)
        # # replace spaces with pipes to create a sentence ready to be used for forced alignment
        # tr2 = '|'.join(tr_list3)
        # transcripts.append(tr2)
          
        # short version of above code 
        tr = f.readline().strip().upper().translate({ord(i): None for i in '?!".,;:[]()—{}/\\'}) \
                                         .translate({ord(i): ' ' for i in '—-'}) \
                                         .translate({ord(i): 'E' for i in 'ÊÉ'}) \
                                         .replace('Æ', 'AE') \
                                         .replace('Œ', 'OE')
        _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
        tr = _RE_COMBINE_WHITESPACE.sub(" ", tr).strip()
        tr_list = tr.split(' ')
        transcripts.append('|'.join(_flatten(list(map(try_convert_to_num, tr_list)))))
        
    return transcripts


def get_speech_data_lists(dirpath, filenames):
    """Gets the speech audio files paths and the transcripts from a single leaf folder in LibriTTS format.

    Args:
      dirpath (str):
        The path to the directory containing the audio files and transcript files.
      filenames (list of str elements):
        the list of all files found by os.walk in this directory.

    Returns:
      speech_files (str, list):
        A sorted list of speech file paths found in this directory.
      transcripts (str, list):
        A list of transcript strings corresponding to each speech file.
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
        elif filename.endswith('.txt'):
            transcripts_files.append(os.path.join(dirpath, filename))

    # check if it is a leaf folder
    if len(transcripts_files):
        transcripts_files.sort()
        speech_files.sort()
        transcripts = convert_transcripts(transcripts_files)
        
    return speech_files, transcripts
