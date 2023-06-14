import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, current + '/..')
import logging
import argparse
from Tools import utils
from config import Config
import nemo_diarization_utils
import pyannote_diarization_utils
import resemblyzer_diarization_utils


def create_speaker_segments(diarization_path, filter_sec=0.0, unified=False):
    # when running diarization with any approach, a corresponding subfolder will be created, e.g.:
    # 'pyannote-diarization/' subfolder will be created and contains a subfolder for each audio recording, each in turn containing a 'diarization.rttm' file.
    # Same for 'resemblyzer-diarization/' and 'nemo-diarization/'.
    # The parent folder of 'X-diarization/' folders should have the original audio recordings wav files.
    for dirpath, subdirs, _ in os.walk(diarization_path, topdown=True):
        # loop only over the subfolders that represent audio files.
        for subdir in subdirs:
            # get rttm file from each subfolder.
            # when using NeMo diarizer output, the path to the RTTM file is slightly different.
            rttm = [f for f in os.listdir(os.path.join(dirpath, subdir, "pred_rttms")) if f.endswith("rttm")] if "nemo-diarization" in dirpath else [f for f in os.listdir(os.path.join(dirpath, subdir)) if f.endswith("rttm")]
            if rttm:
                assert len(rttm) == 1, f"ERROR: there should only be one RTTM file in {os.path.join(dirpath, subdir)}!!!"
                # when using Pyannote, there may be a resemblyzer_preproc_audio/ subfolder, which means the audio for Pyannote diarization was preprocessed in the same way as in Resemblyzer diarization, for fair comparison.
                wav_path = os.path.join(dirpath, "resemblyzer_preproc_audio", subdir + ".wav") if os.path.exists(os.path.join(dirpath, 'resemblyzer_preproc_audio')) else os.path.join("/".join(dirpath.split("/")[:-1]), subdir + ".wav")
                pyannote_diarization_utils.rttm_to_wav(os.path.join(dirpath, subdir, rttm[0]), wav_path, filter_sec=filter_sec, unified=unified)
        break # loop only over the first level of subdirs (where the audio file folders reside).
    logging.info(f"Creation of speaker segments wavs from {diarization_path} complete.")


def pyannote_pipeline(cfg):
    logging.info("Beginning Pyannote diarization.")
    pyannote_diarization_utils.pyannote_diarization(cfg.get("audio_folder"), num_speakers=cfg.get("diarizers/pyannote/num_speakers"), resemblyzer_preprocessing=cfg.get("diarizers/pyannote/resemblyzer_preprocessing"))
    create_speaker_segments(os.path.join(cfg.get("audio_folder"), "pyannote-diarization"), filter_sec=cfg.get("rttm/filter_sec"), unified=cfg.get("rttm/unified"))
    logging.info("Pyannote diarization complete.")


def resemblyzer_pipeline(cfg):
    logging.info("Beginning Resemblyzer diarization.")
    resemblyzer_diarization_utils.resemblyzer_diarization(cfg.get("audio_folder"), similarity_threshold=cfg.get("diarizers/resemblyzer/similarity_threshold"), global_speaker_embeds=cfg.get("diarizers/resemblyzer/global_speaker_embeddings"))
    create_speaker_segments(os.path.join(cfg.get("audio_folder"), "resemblyzer-diarization"), filter_sec=cfg.get("rttm/filter_sec"), unified=cfg.get("rttm/unified"))
    logging.info("Resemblyzer diarization complete.")


def nemo_pipeline(cfg):
    logging.info("Beginning NeMo diarization.")
    nemo_diarization_utils.NemoDiarizer(cfg.get("audio_folder")).nemo_diarization(num_speakers=cfg.get("diarizers/nemo/num_speakers"))
    create_speaker_segments(os.path.join(cfg.get("audio_folder"), "nemo-diarization"), filter_sec=cfg.get("rttm/filter_sec"), unified=cfg.get("rttm/unified"))
    logging.info("NeMo diarization complete.")


def main(cfg):
    utils.mp3_to_wav(cfg.get("audio_folder"))
    utils.audio_preprocessing(cfg.get("audio_folder"), in_place=True)
    
    if cfg.get("diarizers/resemblyzer/include"):
        resemblyzer_pipeline(cfg)
    if cfg.get("diarizers/pyannote/include"):
        pyannote_pipeline(cfg)
    if cfg.get("diarizers/nemo/include"):
        nemo_pipeline(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Splits audio files in a folder into speaker-specific diarized audio snippets.")
    parser.add_argument("config_path", type=str, nargs='?', default=os.getcwd(),
                        help="Path to a yaml config file.")
    # parse command line arguments.
    args = parser.parse_args()
    # create config object from yaml file.
    cfg = Config(args.config_path)

    # setup logging to both console and logfile.
    utils.setup_logging(cfg.get('audio_folder', os.getcwd()), 'diarization.log', console=True, filemode='a')
    # log the command that started the script.
    logging.info(f"Started script via: python {' '.join(sys.argv)}")
    
    main(cfg)
