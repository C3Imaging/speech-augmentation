import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, current + '/..')
import wget
import json
import logging
from tqdm import tqdm
from omegaconf import OmegaConf
import pyannote_diarization_utils
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

SAMPLING_RATE = 16000


class NemoDiarizer():
    def __init__(self, root_path):
        # setup nemo folders
        self.root_path = root_path
        self.output_dir = os.path.join(root_path, 'nemo-diarization')
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir, exist_ok=True)
        self.cfg_dir = os.path.join(self.output_dir, 'nemo-config')
        if not os.path.exists(self.cfg_dir): os.makedirs(self.cfg_dir, exist_ok=True)

        MODEL_CONFIG = os.path.join(self.cfg_dir, 'diar_infer_telephonic.yaml')
        if not os.path.exists(MODEL_CONFIG):
            config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
            MODEL_CONFIG = wget.download(config_url, self.cfg_dir)

        self.config = OmegaConf.load(MODEL_CONFIG)
        # print(OmegaConf.to_yaml(config))

        # for neural MSDD
        self.config.diarizer.msdd_model.model_path = 'diar_msdd_telephonic' # Telephonic speaker diarization model 
        self.config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7, 1.0] # Evaluate with T=0.7 (lower=more generous on speaker overlap) and T=1.0 (no overlap speech)

        # system VAD
        # Here, we use our in-house pretrained NeMo VAD model
        pretrained_vad = 'vad_multilingual_marblenet'
        self.config.diarizer.vad.model_path = pretrained_vad
        self.config.diarizer.vad.parameters.onset = 0.8
        self.config.diarizer.vad.parameters.offset = 0.6
        self.config.diarizer.vad.parameters.pad_offset = -0.05
        self.config.diarizer.oracle_vad = False # compute VAD provided with model_path to vad config

        # speaker embedding model
        pretrained_speaker_model = 'titanet_large'
        self.config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
        self.config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5,1.25,1.0,0.75,0.5] 
        self.config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75,0.625,0.5,0.375,0.1] 
        self.config.diarizer.speaker_embeddings.parameters.multiscale_weights= [1,1,1,1,1] 
        self.config.diarizer.clustering.parameters.oracle_num_speakers = False

        # if not os.path.exists(os.path.join(self.cfg_dir, 'audio_filepaths_list.txt')):
        #     self.create_manifest()

    def create_manifest(self):
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

    def nemo_diarization(self, num_speakers=None):
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
            self.config.diarizer.manifest_filepath = os.path.join(self.cfg_dir, 'input_manifest.json')
            self.config.diarizer.out_dir = subfolder #Directory to store intermediate files and prediction outputs.

            # diarize.
            self.config.num_workers = 0 # workaround to avoid DataLoader ran out of memory error.
            system_vad_msdd_model = NeuralDiarizer(cfg=self.config)
            system_vad_msdd_model.diarize()
            logging.info(f'NeMo diarization: finished diarization of {speech_file}.')

            rttm_path = os.path.join(subfolder, 'pred_rttms', speech_file.split('/')[-1].split('.wav')[0] + ".rttm")
            pyannote_diarization_utils.rttm_to_wav(rttm_path, speech_file, rem_files=True)




