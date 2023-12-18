import os
import sys
import csv
import time
import logging
from pydub import AudioSegment


class Profiler():
    """Generic profiler that can time the execution of a function by calling start before the function call and stop after the function call and outputs the result via the logger."""
    def __init__(self) -> None:
        # log the command that started the script
        logging.info(f"Started script via: python {' '.join(sys.argv)}")
    
    def start(self):
        # start timing how long it takes to run script
        self.start_time = time.perf_counter()

    def stop(self):
        # stop timing and log how long the script took to run
        end_time = time.perf_counter()
        logging.info(f"Finished processing in {time.strftime('%H:%M:%Ss', time.gmtime(end_time - self.start_time))}")


def setup_logging(arg_folder, filename, console=False, filemode='w+'):
    """Set up logging to a logfile and optionally to the console also, if console param is True."""
    if not os.path.exists(arg_folder): os.makedirs(arg_folder, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # track INFO logging events (default was WARNING)
    root_logger.handlers = []  # clear handlers
    root_logger.addHandler(logging.FileHandler(os.path.join(arg_folder, filename), filemode))  # handler to log to file
    root_logger.handlers[0].setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))  # log level and message

    if console:
        root_logger.addHandler(logging.StreamHandler(sys.stdout))  # handler to log to console
        root_logger.handlers[1].setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
 

def BinarySearch(L, x) -> bool:
    """Find leftmost occurrence of value x in sorted list L using binary search and return True if found"""
    first = 0
    last = len(L)-1
    found = False

    while first<=last and not found:
         midpoint = (first + last)//2
         if L[midpoint] == x:
             found = True
         else:
             if x < L[midpoint]:
                 last = midpoint-1
             else:
                 first = midpoint+1

    return found


def dict_to_csv(path, d):
    fnames = [str(key) for key in d.keys()]
    try:
        with open(path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fnames)
            writer.writeheader()
            writer.writerow(d)
    except IOError:
        print("I/O error")


def get_transcript_from_alignment(path):
    transcript = []
    # return the original transcript from an alignment file created by a forced_alignment script
    with open(path, 'r') as f:
        for line in f.readlines():                  
            if "confidence_score" in line:
                pass
            else:
                # append the word
                transcript.append(line.split(",")[1])
    
    return " ".join(transcript)


def flac_to_wav(root_path):
    """in-place conversion of flac files to wav files."""
    for dirpath, _, filenames in os.walk(root_path, topdown=False):
            # get list of speech files from a single folder
            speech_files = []
            # loop through all files found
            for filename in filenames:
                if filename.endswith('.flac'):
                    speech_files.append(os.path.join(dirpath, filename))
            # in place conversion
            if speech_files:
                for speech_file in speech_files:
                    audio = AudioSegment.from_file(speech_file, "flac")
                    audio.export(speech_file.replace("flac", "wav"), format="wav")
                    os.remove(speech_file)


def mp3_to_wav(root_path):
    """in-place conversion of mp3 files to wav files."""
    for dirpath, _, filenames in os.walk(root_path, topdown=True):
        # get list of speech files from a single folder
        speech_files = []
        # loop through all files found
        for filename in filenames:
            if filename.endswith('.mp3'):
                speech_files.append(os.path.join(dirpath, filename))
        # in place conversion
        for speech_file in speech_files:
            audio = AudioSegment.from_mp3(speech_file)
            audio.export(speech_file.replace("mp3", "wav"), format="wav")
            os.remove(speech_file)
            logging.info(f"{speech_file.split('/')[-1].split('.wav')[0]} converted from mp3 to wav.")
        break


def audio_preprocessing(root_path, channels=1, sr=16000, vol_db=0, vol_norm=False, in_place=False):
    """Apply different generic audio preprocessing steps to wav audio files in a folder, as described by the function parameters.

    Args:
      root_path (str):
        The path to the directory containing the audio files.
      channels (int):
        the number of channels the output audio should have.
      sr (int):
        the sampling rate the output audio should have.
      vol_db (int):
        the dB attenuation to apply to the original audio.
      vol_norm (bool):
        whether to normalize the volume with librosa. Use this option if you don't know the dB attenuation needed.
      in_place (bool):
        if True, replace the original audio file with the new version, otherwise save new as a separate file.
    """
    for dirpath, _, filenames in os.walk(root_path, topdown=True):
        # get list of speech files from a single folder.
        speech_files = []
        # loop through all files found.
        for filename in filenames:
            if filename.endswith('.wav'):
                speech_files.append(os.path.join(dirpath, filename))
        # the augmentations
        for speech_file in speech_files:
            modifications = False # flag to specify whether any modifications have been made to the original audio file.
            audio = AudioSegment.from_wav(speech_file)
            if audio.channels != channels:
                assert channels in range(1,3), "ERROR: channels arg should only be set as 1 or 2!"
                audio = audio.set_channels(channels)
                modifications = True
            if audio.frame_rate != sr:
                audio = audio.set_frame_rate(sr)
                modifications = True
            if vol_db:
                audio = audio + vol_db # e.g. 15 for volume boost
                modifications = True

            # w1 = wave.open(speech_file)
            # print("Number of channels is: ",    w1.getnchannels())
            # print("Sample width in bytes is: ", w1.getsampwidth())
            # print("Framerate is: ",             w1.getframerate())
            # print("Number of frames is: ",      w1.getnframes())

            # from pydub import AudioSegment
            # song = AudioSegment.from_file(speech_file, format="wav")
            # song = AudioSegment.from_file("/workspace/datasets/test/yt.m4a")
            # print(song.frame_rate)
            # print(song.channels)

            if modifications:
                if in_place:
                    audio.export(speech_file, format="wav")
                    logging.info(f"{speech_file} has been modified in place with following modifications: n_channels={channels}, sampling_rate={sr}, volume_boost={vol_db}.")
                else:
                    f = speech_file.split(".wav")[0] 
                    f = f + "__dual" if channels == 2 else f + "__mono"
                    f = f + f"_{sr}hz"
                    f = f + f"_plus{vol_db}db" if vol_db >= 0 else f + f"_minus{vol_db}db"
                    f = f + ".wav"
                    audio.export(f, format="wav")
                    logging.info(f"Modified version of {speech_file} saved as {f}.")
            break # loop only over top folder
        
        # workaround if need to normalize audio in-place using librosa due to AudioSegment not being convertible to wav format easily, so must loop through the saved wav files again.
        if vol_norm:
            from scipy.io import wavfile
            from resemblyzer.audio import normalize_volume
            for dirpath, _, filenames in os.walk(root_path, topdown=True):
                # get list of speech files from a single folder.
                speech_files = []
                # loop through all files found.
                for filename in filenames:
                    if filename.endswith('.wav'):
                        speech_files.append(os.path.join(dirpath, filename))
                for speech_file in speech_files:
                    sr, wav = wavfile.read(speech_file)
                    wavfile.write(speech_file.split('.wav')[0]+"test.wav", sr, wav)
                    wav = normalize_volume(wav, -30, increase_only=True)
                    wavfile.write(speech_file, sr, wav)
                    logging.info(f"Volume of {speech_file} normalized.")
                break # loop only over top folder.

