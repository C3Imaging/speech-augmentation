import os
import shlex
import librosa
import subprocess
import numpy as np
import pandas as pd
from scipy.io import wavfile


def pyannote_diarization(root_path):
    """Create RTTM files using pyannote speaker diarization model.
    Args:
        root_path (str):
            The path to a folder containing wav audio files."""

    # 1. visit hf.co/pyannote/speaker-diarization and accept user conditions
    # 2. visit hf.co/pyannote/segmentation and accept user conditions
    # 3. visit hf.co/settings/tokens to create an access token
    # 4. instantiate pretrained speaker diarization pipeline

    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                        use_auth_token="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")

    for dirpath, _, filenames in os.walk(root_path, topdown=True):
        # get list of speech files from a single folder
        speech_files = []
        # loop through all files found
        for filename in filenames:
            if filename.endswith('.wav'):
                speech_files.append(os.path.join(dirpath, filename))
        break # loop only through files in topmost folder

    # create new subfolder for the diarization results
    out_path = os.path.join(root_path, "pyannote-diarization")
    if not os.path.exists(out_path): os.makedirs(out_path, exist_ok=True)

    for speech_file in speech_files:
        # apply the pipeline to an audio file
        diarization = pipeline(speech_file, num_speakers=2)
        # create a separate subfolder for the diarization results for each audio file
        subfolder = os.path.join(out_path, speech_file.split("/")[-1].split(".wav")[0])
        if not os.path.exists(subfolder): os.makedirs(subfolder, exist_ok=True)
        # dump the diarization output to disk using RTTM format
        with open(os.path.join(subfolder, "diarization.rttm"), "w") as rttm:
            diarization.write_rttm(rttm)


def rttm_to_wav(root_path):
    """Segment a wav audio file according to the speakers in an RTTM file for that audio file.

    Args:
        root_path (str):
            The path to a 'pyannote-diarization/' subfolder, containing subfolders for each audio recording, each containing a 'diarization.rttm' file.
            NOTE: The parent folder of 'pyannote-diarization/' should have the audio recordings wav files.
    """

    for dirpath, subdirs, _ in os.walk(root_path, topdown=True):
        for subdir in subdirs:
            # read rttm file into a dataframe
            df = pd.read_csv(os.path.join(dirpath, subdir, "diarization.rttm"), delim_whitespace=True, header=None)
            # manually add header fields according to description in: https://github.com/nryant/dscore#rttm
            df.columns = ["Type", "File ID", "Channel ID", "Turn Onset", "Turn Duration", "Orthography Field", "Speaker Type", "Speaker Name", "Confidence Score", "Signal Lookahead Time"]
            speakers = df["Speaker Name"].unique().tolist() # list of unique speakers
            # create a subfolder for each speaker, where audio snippets will be stored
            speaker_folders = []
            for speaker in speakers:
                subfolder = os.path.join(dirpath, subdir, speaker)
                if not os.path.exists(subfolder): os.makedirs(subfolder, exist_ok=True)
                speaker_folders.append(subfolder)
            # initialise a dict with a count of utterances per speaker
            speakers_dict = dict()
            for speaker in speakers:
                speakers_dict[speaker] = 0
            # loop through rows in df
            for start_time, duration, speaker in zip(df["Turn Onset"], df["Turn Duration"], df["Speaker Name"]):
                in_audio_path = os.path.join("/".join(dirpath.split("/")[:-1]), subdir + ".wav")
                out_dir = os.path.join(dirpath, subdir, speaker)
                out_audio_path = os.path.join(out_dir, speaker + "_" + str(speakers_dict[speaker]) + ".wav")
                subprocess.run(shlex.split(f"ffmpeg -ss {start_time} -i {in_audio_path} -t {duration} {out_audio_path}"))
                speakers_dict[speaker]+=1
            # create a unified audio file for each speaker
            for speaker_folder in speaker_folders:
                combine_wavs(speaker_folder)
        break # loop only over the first level subfolders of root_path, which were created per audio file in the parent folder of root_path.


def combine_wavs(folder_path, sr_out=16000):
    for dirpath, _, filenames in os.walk(folder_path, topdown=True):
        # get list of speech files from a single folder
        speech_files = []
        # loop through all files found
        for filename in filenames:
            if filename.endswith('.wav') and "unified" not in filename:
                speech_files.append(os.path.join(dirpath, filename))
        speech_files.sort()
        # append each wav file to a list
        out_wav = list()
        for speech_file in speech_files:
            wav, sr = librosa.load(speech_file, sr=None)
            if sr != sr_out:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=sr_out)
            out_wav.append(wav)
        break # loop only over the files in the topmost folder
    # collapse list of wavs into a 1D array
    out_wav = np.concatenate(out_wav).ravel()
    wavfile.write(os.path.join(dirpath, "unified.wav"), sr_out, out_wav)


            




