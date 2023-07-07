import os
import json
import wget
import shlex
import logging
import librosa
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config
from typing import Any, List
from scipy.io import wavfile
from omegaconf import OmegaConf
from scipy.io.wavfile import write
from abc import ABC, abstractmethod
from collections import defaultdict


# abstract class for all diarization implementations.
class BaseDiarizer(ABC):
    # show what instance attributes should be defined.
    diarizer: Any # different diarizers do not have a common interface.
    sampling_rate = 16000
    
    @abstractmethod
    def diarize(self, cfg: Config) -> None:
        """Generates a list of transcripts by decoding the output batch of a wav2vec2 ASR acoutic model."""


# Pyannote diarization implementation class.
class PyannoteDiarizer(BaseDiarizer):
    def __init__(self, cfg: Config) -> None:
        from pyannote.audio import Pipeline
        self.cfg = cfg
        self.diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")
    
    def diarize(self):
        from resemblyzer import preprocess_wav
        # get required cfg variables.
        root_path = self.cfg.get("audio_folder") # the path to a folder containing wav audio files.
        num_speakers = self.cfg.get("diarizers/pyannote/num_speakers") # if the number of speakers is explicitly known in advance, otherwise=None (falsy) then number of speakers will be automatically determined by Pyannote.
        resemblyzer_preprocessing = self.cfg.get("diarizers/pyannote/resemblyzer_preprocessing") # whether to preprocess wav in the same way Resemblyzer does, for an apples-to-apples comparison of the diarization output of Pyannote vs Resemblzyer (same preprocessing steps).

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
            logging.info("Pyannote diarization: Resemblyzer-style audio preprocessing activated for Pyannote diarization pipeline.")

        for speech_file in tqdm(speech_files, total=len(speech_files), unit=" audio files", desc=f"Pyannote diarization: processing audio files in {dirpath}, so far"):
            # apply the pipeline to an audio file (input can only be filepath, not ndarray)
            if resemblyzer_preprocessing:
                wav = preprocess_wav(speech_file)
                wav_filename = speech_file.split('/')[-1]
                #  save the Resemblyzer-style preprocessed wav to disk.
                write(os.path.join(resemblyzer_preproc_path, wav_filename), self.sampling_rate, wav)
                diarization = self.diarizer(os.path.join(resemblyzer_preproc_path, wav_filename), num_speakers=num_speakers) if num_speakers else self.diarizer(os.path.join(resemblyzer_preproc_path, wav_filename))
            else:
                diarization = self.diarizer(speech_file, num_speakers=num_speakers) if num_speakers else self.diarizer(speech_file)
            # create a separate subfolder for the diarization results for each audio file
            subfolder = os.path.join(out_path, speech_file.split("/")[-1].split(".wav")[0])
            if not os.path.exists(subfolder): os.makedirs(subfolder, exist_ok=True)
            # dump the diarization output to disk using RTTM format
            with open(os.path.join(subfolder, subfolder.split('/')[-1]+'.rttm'), "w") as rttm:
                diarization.write_rttm(rttm)
                logging.info(f"Pyannote diarization: RTTM file {os.path.join(subfolder, subfolder.split('/')[-1]+'.rttm')} created.")


# Resemblyzer diarization implementation class.
class ResemblyzerDiarizer(BaseDiarizer):
    def __init__(self, cfg: Config) -> None:
        from resemblyzer import VoiceEncoder
        self.cfg = cfg
        self.diarizer = VoiceEncoder("cpu")

    def diarize(self):
        from resemblyzer import preprocess_wav
        # get required cfg variables.
        root_path = self.cfg.get("audio_folder") # the path to a folder containing wav audio files.
        similarity_threshold = self.cfg.get("diarizers/resemblyzer/similarity_threshold")
        # NOTE: if global_speaker_embeds=True, 'root_path' dir must contain a 'speaker-samples/global-speaker-samples/' subdir that contains '<speakerID>.wav audio files, one per speaker, from which speaker embeddings will be created.
        global_speaker_embeds = self.cfg.get("diarizers/resemblyzer/global_speaker_embeddings")

        if global_speaker_embeds:
            speaker_names, speaker_wavs = self._get_speaker_samples(os.path.join(root_path, "speaker-samples", "global-speaker-samples"))
            # Get speaker embeddings
            speaker_embeds = [self.diarizer.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
            logging.info("Resemblyzer diarization: Global speaker embeddings created.")

        for dirpath, _, filenames in os.walk(root_path, topdown=True):
            # get list of speech files from a single folder
            speech_files = []
            # loop through all files found
            for filename in filenames:
                if filename.endswith('.wav'):
                    speech_files.append(os.path.join(dirpath, filename))
            break # loop only through files in topmost folder

        # create new subfolder for the diarization results
        out_path = os.path.join(root_path, "resemblyzer-diarization")
        if not os.path.exists(out_path): os.makedirs(out_path, exist_ok=True)

        # main loop
        for speech_file in tqdm(speech_files, total=len(speech_files), unit=" audio files", desc=f"Resemblyzer diarization: processing audio files in {dirpath}, so far"):
            # apply the pipeline to an audio file
            # create a separate subfolder for the diarization results for each audio file
            subfolder = os.path.join(out_path, speech_file.split("/")[-1].split(".wav")[0])
            if not os.path.exists(subfolder): os.makedirs(subfolder, exist_ok=True)
            logging.info(f"Resemblyzer diarization: Populating {subfolder} with diarization wavs based on {speech_file}.")

            # preprocesses audio into 16kHz mono-channel, removes long silences, normalises audio volume
            wav = preprocess_wav(speech_file)

            if global_speaker_embeds:
                wav, similarity_dict, wav_splits = self._get_predictions(wav, speaker_names, speaker_embeds)
            else:
                speaker_folder = os.path.join(root_path, "speaker-samples", speech_file.split("/")[-1].split(".wav")[0])
                speaker_names, speaker_wavs = self._get_speaker_samples(speaker_folder)
                # Get speaker embeddings
                speaker_embeds = [self.diarizer.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
                wav, similarity_dict, wav_splits = self._get_predictions(wav, speaker_names, speaker_embeds)

            speaker_segments = self._get_speaker_segments(similarity_dict, wav_splits, similarity_threshold)

            # initialise empty list per speaker.
            # elements of a list will be the frame indexes of the original audio file where that speaker has a confidence value above the thrseshold.
            speakers_frames_idxs_dict = dict()
            for speaker in speaker_names:
                speakers_frames_idxs_dict[speaker] = list()

            # add the indexes of frames (a list) determined to be the speaker speaking to the corresponding speaker key (will be a list of lists, where each lowest-level list is a segment of frame indexes)
            for speaker, seg in zip(speaker_segments, wav_splits):
                if speaker != 'none':
                    speakers_frames_idxs_dict[speaker].append(np.arange(seg.start,seg.stop))
            # collapse each speaker's lists into a 1D array of sorted unique frame indexes
            for k, v in speakers_frames_idxs_dict.items():
                if v:
                    speakers_frames_idxs_dict[k] = np.unique(np.concatenate(v).ravel())
                else:
                    logging.info(f"Resemblyzer diarization: {subfolder} has no diarized audio for speaker {k}.")
                    # out_wav = os.path.join(subfolder, f"{k}_unified_confthresh_{str(similarity_threshold)}.wav")
                    # # pick out all frames (by index) from wav file that belong to speaker k.
                    # write(out_wav, SAMPLING_RATE, wav[speakers_frames_idxs_dict[k].tolist()])
                    # logging.info(f"Resemblyzer diarization: {out_wav} created.")
            # delete any speakers that do not have any spoken segments
            for k in list(speakers_frames_idxs_dict.keys()):
                # hack: cast list to numpy array to check if its empty
                v = np.array(speakers_frames_idxs_dict[k])
                if not v.any():
                    del speakers_frames_idxs_dict[k]
            # initialise empty list per speaker.
            # value per speaker will be a list of tuples, where each tuple is the start and stop frame index of a continuous segment of frames spoken by that speaker.
            speaker_wavs = dict()
            for speaker in speakers_frames_idxs_dict.keys():
                speaker_wavs[speaker] = list()

            # loop through all frame indexes of each speaker
            for k, v in speakers_frames_idxs_dict.items():
                seg_start = seg_stop = v[0] # indexes/frames
                for i in v:
                    # stopping condition -> gap in continuous sequence detected
                    if i > seg_stop + 1:
                        # save frames indexes seg_start and seg_stop (included) as tuple
                        speaker_wavs[k].append((seg_start,seg_stop))
                        # reset seg_start and seg_stop
                        seg_start = i
                    seg_stop = i

            # combine segments where the gap between them is less than 'tolerance' number of frames
            tolerance = 50

            def join_speaker_segments(speaker_wavs_dict, tolerance):
                for k, v in speaker_wavs_dict.items():
                    for i in range(0, len(v)-1):
                        # gap = v[i+1][0] - v[i][1] - 1
                        # if the gap is less than or equal to the tolerance allowable, combine:
                        if v[i+1][0] <= v[i][1] + tolerance + 1:
                            speaker_wavs_dict[k][i] = (v[i][0], v[i+1][1]) # replace (i)'th tuple with combined
                            del speaker_wavs_dict[k][i+1] # delete (i+1)'st tuple
                            return True
                return False

            while join_speaker_segments(speaker_wavs, tolerance): pass

            # # filter out segments shorter than filter_sec in length (in seconds) so they don't get included in the created RTTM file.
            # filter_sec = 1.0
            # if filter_sec > 0.0:
            #     for k, v in speaker_wavs.items(): speaker_wavs[k] = filter(lambda tup: float(tup[1]-tup[0]+1)/SAMPLING_RATE >= filter_sec, v)    

            # create RTTM file
            COLUMN_NAMES=['Type','File ID','Channel ID','Turn Onset','Turn Duration','Orthography Field','Speaker Type', 'Speaker Name', 'Confidence Score', 'Signal Lookahead Time']
            df = pd.DataFrame(columns=COLUMN_NAMES, index=[0])
            for k, v in speaker_wavs.items():
                for tup in v:
                    df.loc[len(df.index)] = ['SPEAKER', subfolder.split('/')[-1], 1, float(tup[0]/self.sampling_rate), float(tup[1]-tup[0]+1)/self.sampling_rate, '<NA>', '<NA>', k, '<NA>', '<NA>'] 
            # RTTM filename is same as original wav filename
            df.to_csv(os.path.join(subfolder, subfolder.split('/')[-1]+'.rttm'), sep =' ', header=False, index=False)
            logging.info(f"Resemblyzer diarization: RTTM file {os.path.join(subfolder, subfolder.split('/')[-1]+'.rttm')} created.")

            # write(out_wav, SAMPLING_RATE, wav[seg_start:seg_stop+1])
            # logging.info(f"Resemblyzer diarization: Population of {subfolder} with diarization wavs complete.")

    def _get_speaker_samples(self, root_path):
        """Returns example wav per speaker from which speaker embeddings will be made by Voice Encoder model.

        Args:
        root_path (str):
            The path to a folder contianing wav audio files and a 'speaker-samples/' subdir.
        
        Returns:
        speaker_names (list, str):
            speaker IDs.
        speaker_wavs (list, np.ndarray[int, float]):
            wav data per speaker.
        """
        # load example speech for speakers, from which speaker embeddings will be created
        assert os.path.exists(root_path), f"ERROR: {root_path} does not exist!!! Please manually create it."

        speaker_names = []
        speaker_wavs = []
        for dirpath, _, filenames in os.walk(root_path, topdown=True):
            # get list of speech files from a single folder
            speech_files = []
            # loop through all files found
            for filename in filenames:
                if filename.endswith('.wav'):
                    speech_files.append(os.path.join(dirpath, filename))
            break # loop only through files in topmost folder
        # loop through all files found
        assert len(speech_files), f"ERROR: no wav files found in {root_path}!!! Please manually add example speech wav files for each speaker."
        for speech_file in speech_files:
            # sr, wav = wavfile.read(os.path.join(dirpath, speech_file))
            wav, sr = librosa.load(os.path.join(dirpath, speech_file), sr=None)
            if sr != self.sampling_rate:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sampling_rate)
            speaker_wavs.append(wav)
            speaker_name = speech_file.split(".wav")[0].split("/")[-1]
            speaker_names.append(speaker_name)
        return speaker_names, speaker_wavs

    def _get_predictions(self, wav, speaker_names, speaker_embeds):
        """Voice Encoder model calculates speaker embeddings and similarity score predictions per speaker for each segment of audio.

        Args:
        wav (list, [int, float]):
            a multispeaker audio wav.
        speaker_names (list, str):
            speaker IDs.
        speaker_embeds (list, ndarray[float]):
            a fixed-length speaker embedding vector for each speaker.
        
        Returns:
        wav (list, [int, float]):
            A multispeaker audio wav, possibly padded.
        similarity_dict (dict, {str: np.ndarray}):
            keys are speaker IDs, values are similarity scores for each audio segment as cut by wav_splits.
        wav_splits (list, slice):
            start and stop frame indexes of each audio segment.
        """
        ## Compare speaker embeds to the continuous embedding of the interview
        # Derive a continuous embedding of the interview. We put a rate of 16, meaning that an 
        # embedding is generated every 0.0625 seconds. It is good to have a higher rate for speaker 
        # diarization, but it is not so useful for when you only need a summary embedding of the 
        # entire utterance. A rate of 2 would have been enough, but 16 is nice for the sake of the 
        # demonstration. 
        # We'll exceptionally force to run this on CPU, because it uses a lot of RAM and most GPUs 
        # won't have enough. There's a speed drawback, but it remains reasonable.
        logging.info(f"Resemblyzer diarization: Creating embedding for entire wav file.")
        # dividing wav into segments of multiple frames every 0.0625 seconds.
        # each segment will have an embedding of length 256 (same length as speaker embeddings).
        _, cont_embeds, wav_splits = self.diarizer.embed_utterance(wav, return_partials=True, rate=16)
        # pad the wav file with zeros, since embed_utterance() did for its segments.
        if wav_splits[-1].stop > len(wav) - 1:
            wav = np.pad(wav, (0, wav_splits[-1].stop - len(wav)), "constant")
        # Get the continuous similarity for every speaker. It amounts to a dot product between the 
        # embedding of the speaker and the continuous embedding of the interview.
        # for each speaker, for each segment, a similarity score value will be produced in the range [0.0,1.0]
        similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in 
                        zip(speaker_names, speaker_embeds)}
        
        return wav, similarity_dict, wav_splits

    def _get_speaker_segments(self, similarity_dict, wav_splits, similarity_threshold):
        """Returns a list of most likely speaker IDs per segment.

        Args:
        similarity_dict (dict, {str: np.ndarray}):
            keys are speaker IDs, values are similarity scores for each audio segment as cut by wav_splits.
        wav_splits (list, slice):
            start and stop frame indexes of each audio segment.
        similarity_threshold (float):
            minimum confidence threshold for considering a speech segment to be confidently spoken by a speaker.
        
        Returns:
        speaker_segments (list, str):
            most likely speaker ID per segment.
        """
        # a speaker per segment, 'none' if segment has similarity score below similarity_threshold for all known speakers.
        speaker_segments = list()
        # loop through each segment
        for i in range(len(wav_splits)):
            # get speaker with max similarity
            similarities = [s[i] for s in similarity_dict.values()]
            best = np.argmax(similarities)
            name, similarity = list(similarity_dict.keys())[best], similarities[best]
            speaker_segments.append(name) if similarity > similarity_threshold else speaker_segments.append('none')
        return speaker_segments

# Resemblyzer-specific utility function.
def create_speaker_wavs(wav_path, speaker_names, start_stop_times):
    """Creates wav files per speaker, containing example speech from which speaker embeddings will be created by the Resemblyzer Encoder model.
    If you want to create a 'global-speaker-samples/' dir, just rename the resultant created folder.
    
    Args:
    wav_path (str):
        path to the multispeaker wav file from which to extract speaker wavs.
    speaker_names (list str):
        explicit list of speaker names in order.
    start_stop_times (list float):
        list of pairs of start-stop times per speaker as a 1D array.
    """
    SAMPLING_RATE = 16000
    # create a subfolder for that wav audio file
    # create a separate subfolder for the diarization results for each audio file
    subfolder = os.path.join('/'.join(wav_path.split("/")[:-1]), "speaker-samples", wav_path.split("/")[-1].split(".wav")[0])
    if not os.path.exists(subfolder): os.makedirs(subfolder, exist_ok=True)
    wav, sr = librosa.load(wav_path, sr=None)
    if sr != SAMPLING_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLING_RATE)
    # TODO: try normalizing the speaker wavs too, just like the target audio file to diarize.
    # wav = audio.normalize_volume(wav, -30, increase_only=True)
    
    # Cut some segments from single speakers as reference audio.
    # Encoder model will later create speaker embeddings from manually selected segments of example speech from each speaker from a single audio file.
    # a segment representing a speaker = [start time in seconds, end time in seconds]
    for speaker_name, times in zip(speaker_names, [start_stop_times[i:i + 2] for i in range(0, len(start_stop_times), 2)]):
        speaker_wav = wav[int(times[0] *SAMPLING_RATE):int(times[1] * SAMPLING_RATE)]
        write(os.path.join(subfolder, f"{speaker_name}.wav"), SAMPLING_RATE, speaker_wav)


# Nemo diarization implementation class.
class NemoDiarizer(BaseDiarizer):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # setup nemo folders.
        self.root_path =  self.cfg.get("audio_folder") # the path to a folder containing wav audio files.
        self.output_dir = os.path.join(self.root_path, 'nemo-diarization')
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir, exist_ok=True)
        self.cfg_dir = os.path.join(self.output_dir, 'nemo-config')
        if not os.path.exists(self.cfg_dir): os.makedirs(self.cfg_dir, exist_ok=True)

        MODEL_CONFIG = os.path.join(self.cfg_dir, 'diar_infer_telephonic.yaml')
        if not os.path.exists(MODEL_CONFIG):
            config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
            MODEL_CONFIG = wget.download(config_url, self.cfg_dir)

        self.model_config = OmegaConf.load(MODEL_CONFIG)
        # print(OmegaConf.to_yaml(config))

        # for neural MSDD.
        self.model_config.diarizer.msdd_model.model_path = 'diar_msdd_telephonic' # Telephonic speaker diarization model.
        self.model_config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7, 1.0] # Evaluate with T=0.7 (lower=more generous on speaker overlap) and T=1.0 (no overlap speech).

        # system VAD.
        # Here, we use our in-house pretrained NeMo VAD model.
        pretrained_vad = 'vad_multilingual_marblenet'
        self.model_config.diarizer.vad.model_path = pretrained_vad
        self.model_config.diarizer.vad.parameters.onset = 0.8
        self.model_config.diarizer.vad.parameters.offset = 0.6
        self.model_config.diarizer.vad.parameters.pad_offset = -0.05
        self.model_config.diarizer.oracle_vad = False # compute VAD provided with model_path to vad config.

        # speaker embedding model.
        pretrained_speaker_model = 'titanet_large'
        self.model_config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
        self.model_config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5,1.25,1.0,0.75,0.5] 
        self.model_config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75,0.625,0.5,0.375,0.1] 
        self.model_config.diarizer.speaker_embeddings.parameters.multiscale_weights= [1,1,1,1,1] 
        self.model_config.diarizer.clustering.parameters.oracle_num_speakers = False

        # if not os.path.exists(os.path.join(self.cfg_dir, 'audio_filepaths_list.txt')):
        #     self._create_manifest()

    def diarize(self):
        from nemo.collections.asr.models.msdd_models import NeuralDiarizer
        # get required cfg variables.
        num_speakers = self.cfg.get("diarizers/nemo/num_speakers")

        for dirpath, _, filenames in os.walk(self.root_path, topdown=True):
            # get list of speech files from a single folder.
            speech_files = []
            # loop through all files found.
            for filename in filenames:
                if filename.endswith('.wav'):
                    speech_files.append(os.path.join(dirpath, filename))
            break # loop only through files in topmost folder.

        for speech_file in tqdm(speech_files, total=len(speech_files), unit=" audio files", desc=f"NeMo diarization: processing audio files in {dirpath}, so far"):
            logging.info(f'NeMo diarization: starting diarization of {speech_file}.')
            # apply the pipeline to an audio file.
            # create a manifest for input wav file.
            meta = {
                'audio_filepath': speech_file, 
                'offset': 0, 
                'duration':None, 
                'label': 'infer', 
                'text': '-', 
                'num_speakers': num_speakers, # if None then determines automatically.
                'rttm_filepath': None, 
                'uem_filepath' : None
            }
            # write the manifest file to config dir (will be overwritten by future audio files).
            with open(os.path.join(self.cfg_dir, 'input_manifest.json'), 'w') as fp:
                json.dump(meta,fp)
                fp.write('\n')
            # create a separate subfolder for the diarization results for each audio file.
            subfolder = os.path.join(self.output_dir, speech_file.split("/")[-1].split(".wav")[0])
            if not os.path.exists(subfolder): os.makedirs(subfolder, exist_ok=True)

            # set some current audio file specific config info.
            self.model_config.diarizer.manifest_filepath = os.path.join(self.cfg_dir, 'input_manifest.json')
            self.model_config.diarizer.out_dir = subfolder #Directory to store intermediate files and prediction outputs.

            # diarize.
            self.model_config.num_workers = 0 # workaround to avoid DataLoader ran out of memory error.
            system_vad_msdd_model = NeuralDiarizer(cfg=self.model_config)
            system_vad_msdd_model.diarize()
            logging.info(f'NeMo diarization: finished creating RTTM file of {speech_file}.')

    def _create_manifest(self):
        for dirpath, _, filenames in os.walk(self.root_path, topdown=True):
            # get list of speech files from a single folder.
            speech_files = []
            # loop through all files found.
            for filename in filenames:
                if filename.endswith('.wav'):
                    speech_files.append(os.path.join(dirpath, filename))
            break # loop only through files in topmost folder.
        
        # create file with a list of all the audio files to diarize.
        with open(os.path.join(self.cfg_dir, 'audio_filepaths_list.txt'), 'w') as fp:
            for speech_file in speech_files:
                fp.write(speech_file)
                fp.write('\n')


# common utility functions for diarization task.
def rttm_to_wav(rttm_path, wav_path, sr_out=16000, filter_sec=0.0, unified=False):
    """Segment a wav audio file according to the speakers in an RTTM file for that audio file.

    Args:
        rttm_path (str):
            path to a RTTM file.
        wav_path (str):
            path to the multispeaker wav file to segment.
        sr_out (int):
            output sampling rate for created speaker segments audio files.
        filter_sec (float):
            do not include speaker audio segments from the RTTM file that are less than 'filter_sec' seconds in length.
        unified (bool):
            if True, will create a unified speaker audio file from all the segments for that speaker.
    """
    logging.info(f"Processing RTTM file {rttm_path} to split {wav_path} into seperate speaker files.")
    # read rttm file into a dataframe.
    try:
        df = pd.read_csv(rttm_path, delim_whitespace=True, header=None)
        # manually add header fields according to description in: https://github.com/nryant/dscore#rttm
        df.columns = ["Type", "File ID", "Channel ID", "Turn Onset", "Turn Duration", "Orthography Field", "Speaker Type", "Speaker Name", "Confidence Score", "Signal Lookahead Time"]
        speakers = df["Speaker Name"].unique().tolist() # list of unique speakers.
        # create a subfolder for each speaker, where audio snippets will be stored.
        speaker_folders = []
        for speaker in speakers:
            subfolder = os.path.join('/'.join(rttm_path.split('/')[:-1]), speaker)
            if not os.path.exists(subfolder): os.makedirs(subfolder, exist_ok=True)
            speaker_folders.append(subfolder)
        # initialise a dict with a count of utterances per speaker.
        speakers_dict = dict()
        for speaker in speakers:
            speakers_dict[speaker] = 0
        # loop through rows in df.
        for start_time, duration, speaker in zip(df["Turn Onset"], df["Turn Duration"], df["Speaker Name"]):
            out_dir = os.path.join('/'.join(rttm_path.split('/')[:-1]), speaker)
            out_audio_path = os.path.join(out_dir, rttm_path.split('/')[-2] + "_" + speaker + "_" + str(speakers_dict[speaker]) + ".wav")
            # filter out any segments of speech that are shorter than the minimum allowed length in seconds.
            if duration >= filter_sec:
                subprocess.run(shlex.split(f"ffmpeg -y -ss {start_time} -i {wav_path} -t {duration} {out_audio_path}"))
                logging.info(f"{out_audio_path} created.")
                speakers_dict[speaker]+=1
        # create a unified audio file for each speaker.
        if unified:
            for speaker_folder in speaker_folders:
                combine_wavs(speaker_folder, sr_out=sr_out)
        logging.info(f"Processing of RTTM file {rttm_path} complete.")
    except pd.errors.EmptyDataError:
        logging.info(f"{rttm_path} is empty. No speaker segments will be created from {wav_path}")


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
    if len(out_wav):
        # collapse list of wavs into a 1D array
        out_wav = np.concatenate(out_wav).ravel()
        wavfile.write(os.path.join(dirpath, "unified.wav"), sr_out, out_wav)
    else:
        logging.info(f"No wav files to concatenate in {dirpath}.")
    
    logging.info(f"Combining wav files in {folder_path} complete.")

# class for evaluating the quality of different diarizers.
class Diarization_Eval():
    def __init__(self, gt_rttms: List[str], hyp_rttms: List[str]) -> None:
        self.gt_rttms = gt_rttms # list of paths to ground truth RTTM files.
        self.hyp_rttms = hyp_rttms # list of paths to hypothesis (predicted) RTTM files.
        self.gt_rttms.sort()
        self.hyp_rttms.sort()
        # assert filenames are the same across the ground-truth and hypothesis RTTM files.
        assert list(map(lambda x: x.split('/')[-1].split('.rttm')[0], gt_rttms)) == list(map(lambda x: x.split('/')[-1].split('.rttm')[0], hyp_rttms)), "ERROR: The ground truth and hypothesis RTTM files don't match!!!"

        # load all RTTM files as corresponding lists of dataframes and initialise them as instance variables.
        self._init_dfs()
        
    def _init_dfs(self):
        """Initialises two lists: corresponding ground-truth RTTM files and hypothesis RTTM files initialised as dataframe objects.
        The hypotheses are created by a diarizer on the same diarization evaluation dataset as the ground truth."""
        gt_dfs = list()
        hyp_dfs = list()
        COLUMN_NAMES=['Type','File ID','Channel ID','Turn Onset','Turn Duration','Orthography Field','Speaker Type', 'Speaker Name', 'Confidence Score', 'Signal Lookahead Time']
        for gt_rttm, hyp_rttm in zip(self.gt_rttms, self.hyp_rttms):
            # load ground truth and hypothesis RTTM files into dataframes.
            gt_df = pd.read_csv(gt_rttm, delim_whitespace=True, header=None)
            hyp_df = pd.read_csv(hyp_rttm, delim_whitespace=True, header=None)
            # set column names as per RTTM documentation for more explicit indexing.
            gt_df.columns = COLUMN_NAMES
            hyp_df.columns = COLUMN_NAMES
            gt_dfs.append(gt_df)
            hyp_dfs.append(hyp_df)

        self.gt_dfs, self.hyp_dfs = gt_dfs, hyp_dfs
    
    def _calculate_edgecase_errors(self) -> float:
        """Runs multiple edge case scenarios as a preprocessing/prefiltering stage before running calculate_error.
        Returns 1.0 if an edge case happens, meaning error is maximum, else returns 0.0, meaning no edge case happened."""

        # get the set of speakers from both dataframes by explicit indexing.
        gt_speakers_set = set(self.gt_df['Speaker Name'].tolist())
        hyp_speakers_set = set(self.hyp_df['Speaker Name'].tolist())

        # edge case 1: hypothesis has none of the ground truth speakers, therefore maximum error (1.0).
        if len(gt_speakers_set - hyp_speakers_set) == len(gt_speakers_set):
            return 1.0
        else:
            return 0.0

    def calculate_error(self, evaluation_metric) -> float:
        """Calculates the difference/divergence/error in range [0.0,1.0] between each pair of ground-truth and hypothesis diarizations
        according to a particular implementation metric function passed as an argument and returns the average value."""
        edge_case_error = self._calculate_edgecase_errors()
        if not edge_case_error:
            return evaluation_metric(self.gt_dfs, self.hyp_dfs)
        else:
            return edge_case_error

# different evaluation metrics to run between a list of ground truth RTTM files loaded as dataframes and a corresponding list of hypothesis dataframes, returning the average.
def evalMetric_gtSpeakerTimeCounter(gt_dfs: List[pd.DataFrame], hyp_dfs: List[pd.DataFrame]) -> float:
    """Simplest evaluation metric that gets the difference between the amount of time in seconds for each ground truth speaker.
    Only considers the speakers from the ground truth files and does not punish the diarization model for predicting other speakers.
    The larger the difference, the higher the error."""
    err_avg_accum_total = 0.0 # accumulated average error over all the RTTM pairs.
    # loop through each pair of dfs created from rttms.
    for gt_df, hyp_df in zip(gt_dfs, hyp_dfs):
        # initialise empty dict of tuples per ground truth speaker, where tuple is (total time spoken by speaker in ground truth, total time spoken by same speaker in hypothesis).
        gt_speakers_times = defaultdict(tuple)
        # get set of unique speakers from ground truth df.
        gt_speakers = list(set(gt_df['Speaker Name'].tolist()))
        # loop through the speech segments of each ground truth speaker.
        for gt_speaker in gt_speakers:
            # GROUND TRUTH
            # get a dataframe of all rows (which represent spoken segments where that speaker is speaking) where the speaker name = gt_speaker.
            gt_speaker_df = gt_df.loc[gt_df['Speaker Name'] == gt_speaker]
            # sum up the time durations to get the total time spoken by that speaker.
            gt_speaker_time = gt_speaker_df['Turn Duration'].sum()
            # HYPOTHESIS
            # get a dataframe of all rows (which represent spoken segments where that speaker is speaking) where the speaker name = gt_speaker.
            hyp_speaker_df = hyp_df.loc[hyp_df['Speaker Name'] == gt_speaker]
            # sum up the time durations to get the total time spoken by that speaker.
            hyp_speaker_time = hyp_speaker_df['Turn Duration'].sum()
            # append tuple to dict.
            gt_speakers_times[gt_speaker] = (gt_speaker_time, hyp_speaker_time)
        # calculate the error metric.
        # neeed to create your own gauge of error tolerance, in order to map time differences to error scores in range [0.0,1.0]:
        #       1 second difference = 0.1 error.
        #       2 seconds of difference = 0.2 error.
        #       10 seconds of difference = 1.0 error (max error, just like edge case scenario, meaning completely unacceptable difference).
        #       anything above 10 seconds of difference = 1.0 error also.
        err_accum1 = 0.0
        # loop through pairs of speaker times and accumulate the absolute difference in time between ground truth and hypothesis for each ground truth speaker segment.
        for gt_sec, hyp_sec in gt_speakers_times.values():
            diff = abs(gt_sec - hyp_sec)
            # map time difference to an error value according to the gauge.
            err = 1.0 if diff >= 10.0 else diff
            err_accum1 += e
        # get the average error for this diarizer on all its total speaker time predictions for the ground truth speakers in this RTTM pair. 
        err_avg1 = err_accum1 / len(gt_speakers_times)
        err_avg_accum_total += err_avg1
    # get the average error across the entire evaluation set.
    err_avg_total = err_avg_accum_total / len(gt_dfs)

    return err_avg_total
        



