import os
import shlex
import librosa
import logging
import subprocess
import numpy as np
import pandas as pd
from scipy.io import wavfile
from resemblyzer import preprocess_wav
from scipy.io.wavfile import write


def pyannote_diarization(root_path, num_speakers=0, resemblyzer_preprocessing=False):
    """Create RTTM files using pyannote speaker diarization model.
    Args:
        root_path (str):
            the path to a folder containing wav audio files.
        num_speakers (int):
            if the number of speakers is explicitly known in advance, otherwise=0 (falsy) then number of speakers will be automatically determined by Pyannote.
        resemblyzer_preprocessing (bool):
            whether to preprocess wav in the same way Resemblyzer does, for an apples-to-apples comparison of the diarization output of Pyannote vs Resemblzyer (same preprocessing steps).
    """

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

    # create new subfolder for the diarization results.
    out_path = os.path.join(root_path, "pyannote-diarization")
    if not os.path.exists(out_path): os.makedirs(out_path, exist_ok=True)

    # make subfolder for Resemblyzer-style preprocessed audio.
    if resemblyzer_preprocessing:
        resemblyzer_preproc_path = os.path.join(out_path, 'resemblyzer_preproc_audio')
        if not os.path.exists(resemblyzer_preproc_path): os.makedirs(resemblyzer_preproc_path, exist_ok=True)
        logging.info("Resemblyzer-style audio preprocessing activated for Pyannote diarization pipeline.")

    for speech_file in speech_files:
        # apply the pipeline to an audio file (input can only be filepath, not ndarray)
        if resemblyzer_preprocessing:
            wav = preprocess_wav(speech_file)
            wav_filename = speech_file.split('/')[-1]
            #  save the Resemblyzer-style preprocessed wav to disk.
            write(os.path.join(resemblyzer_preproc_path, wav_filename), 16000, wav)
            diarization = pipeline(os.path.join(resemblyzer_preproc_path, wav_filename), num_speakers=num_speakers) if num_speakers else pipeline(os.path.join(resemblyzer_preproc_path, wav_filename))
        else:
            diarization = pipeline(speech_file, num_speakers=num_speakers) if num_speakers else pipeline(speech_file)
        # create a separate subfolder for the diarization results for each audio file
        subfolder = os.path.join(out_path, speech_file.split("/")[-1].split(".wav")[0])
        if not os.path.exists(subfolder): os.makedirs(subfolder, exist_ok=True)
        # dump the diarization output to disk using RTTM format
        with open(os.path.join(subfolder, "diarization.rttm"), "w") as rttm:
            diarization.write_rttm(rttm)
            logging.info(f'RTTM file {os.path.join(subfolder, "diarization.rttm")} created.')


def rttm_to_wav(root_path, resemblyzer_preprocessing=False):
    """Segment a wav audio file according to the speakers in an RTTM file for that audio file.

    Args:
        root_path (str):
            the path to a 'pyannote-diarization/' subfolder, containing subfolders for each audio recording, each containing a 'diarization.rttm' file.
            NOTE: The parent folder of 'pyannote-diarization/' should have the audio recordings wav files.
        resemblyzer_preprocessing (bool):
            set to True if used Resemblyzer-style preprocessed audio for the saved diarized audio segments (RTTM file).
    """

    for dirpath, subdirs, _ in os.walk(root_path, topdown=True):
        for subdir in subdirs:
            # loop only over the subfolders that represent audio files.
            if 'resemblyzer_preproc_audio' not in subdir: # loop only over speaker subdirs.
                logging.info(f"Processing RTTM file in {subdir}.")
                # read rttm file into a dataframe.
                df = pd.read_csv(os.path.join(dirpath, subdir, "diarization.rttm"), delim_whitespace=True, header=None)
                # manually add header fields according to description in: https://github.com/nryant/dscore#rttm
                df.columns = ["Type", "File ID", "Channel ID", "Turn Onset", "Turn Duration", "Orthography Field", "Speaker Type", "Speaker Name", "Confidence Score", "Signal Lookahead Time"]
                speakers = df["Speaker Name"].unique().tolist() # list of unique speakers.
                # create a subfolder for each speaker, where audio snippets will be stored.
                speaker_folders = []
                for speaker in speakers:
                    subfolder = os.path.join(dirpath, subdir, speaker)
                    if not os.path.exists(subfolder): os.makedirs(subfolder, exist_ok=True)
                    speaker_folders.append(subfolder)
                # initialise a dict with a count of utterances per speaker.
                speakers_dict = dict()
                for speaker in speakers:
                    speakers_dict[speaker] = 0
                # loop through rows in df.
                for start_time, duration, speaker in zip(df["Turn Onset"], df["Turn Duration"], df["Speaker Name"]):
                    in_audio_path = os.path.join(dirpath, "resemblyzer_preproc_audio", subdir + ".wav") if resemblyzer_preprocessing else os.path.join("/".join(dirpath.split("/")[:-1]), subdir + ".wav")
                    out_dir = os.path.join(dirpath, subdir, speaker)
                    out_audio_path = os.path.join(out_dir, speaker + "_" + str(speakers_dict[speaker]) + ".wav")
                    subprocess.run(shlex.split(f"ffmpeg -ss {start_time} -i {in_audio_path} -t {duration} {out_audio_path}"))
                    logging.info(f"{out_audio_path} created.")
                    speakers_dict[speaker]+=1
                # create a unified audio file for each speaker.
                for speaker_folder in speaker_folders:
                    combine_wavs(speaker_folder)
                logging.info(f"Processing of RTTM file in {subdir} complete.")
        break # loop only over the first level subfolders of root_path, which were created per audio file in the parent folder of root_path.


def combine_wavs(folder_path, sr_out=16000):
    logging.info(f"Combining wav files in {folder_path} into a single unified wav file.")
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
    logging.info(f"Combining wav files in {folder_path} complete.")


            




