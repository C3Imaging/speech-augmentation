import os
import sys
import yaml
current = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, current + '/..')
import logging
import argparse
from Utils import utils
from config import Config
import diarization_utils


def create_speaker_segments(diarization_path, filter_sec=0.0, unified=False):
    """parses RTTM file of each audio recording subfolder in 'diarization_path' folder to create speaker segments wav files for each instance of each speaker speaking.
    Creates a speaker subfolder in the audio recording subfolder for each speaker.
    Filters out instances of speech that are shorter than 'filter_sec' seconds in duration."""

    # when running diarization with any approach, a corresponding subfolder will be created, e.g.:
    # 'pyannote-diarization/' subfolder will be created and contains a subfolder for each audio recording, each in turn containing a 'diarization.rttm' file.
    # Same for 'resemblyzer-diarization/' and 'nemo-diarization/' etc.
    # The parent folder of 'X-diarization/' folders should have the original audio recordings wav files.
    for dirpath, subdirs, _ in os.walk(diarization_path, topdown=True):
        # loop only over the subfolders that represent audio files.
        for subdir in subdirs:
            if 'nemo' not in subdir:
                # get rttm file from each subfolder.
                # when using NeMo diarizer output, the path to the RTTM file is slightly different.
                rttm = [f for f in os.listdir(os.path.join(dirpath, subdir, "pred_rttms")) if f.endswith("rttm")] if "nemo-diarization" in dirpath else [f for f in os.listdir(os.path.join(dirpath, subdir)) if f.endswith("rttm")]
                if rttm:
                    assert len(rttm) == 1, f"ERROR: there should only be one RTTM file in {os.path.join(dirpath, subdir)}!!!"
                    # when using Pyannote, there may be a resemblyzer_preproc_audio/ subfolder, which means the audio for Pyannote diarization was preprocessed in the same way as in Resemblyzer diarization, for fair comparison.
                    wav_path = os.path.join(dirpath, "resemblyzer_preproc_audio", subdir + ".wav") if os.path.exists(os.path.join(dirpath, 'resemblyzer_preproc_audio')) else os.path.join("/".join(dirpath.split("/")[:-1]), subdir + ".wav")
                    # nemo diarization has an extra subfolder
                    subdir = os.path.join(subdir, "pred_rttms") if 'nemo' in dirpath else subdir
                    
                    diarization_utils.rttm_to_wav(os.path.join(dirpath, subdir, rttm[0]), wav_path, filter_sec=filter_sec, unified=unified)
        break # loop only over the first level of subdirs (where the audio file folders reside).
    logging.info(f"Creation of speaker segments wavs from {diarization_path} complete.")


def resemblyzer_pipeline(cfg):
    logging.info("Beginning Resemblyzer diarization.")
    d = diarization_utils.ResemblyzerDiarizer(cfg)
    d.diarize()
    create_speaker_segments(os.path.join(cfg.get("audio_folder"), "resemblyzer-diarization"), filter_sec=cfg.get("rttm/filter_sec"), unified=cfg.get("rttm/unified"))
    diarization_utils.rttm_filter(os.path.join(cfg.get("audio_folder"), "resemblyzer-diarization"), filter_sec=cfg.get("rttm/filter_sec"))

    logging.info("Resemblyzer diarization complete.")


def pyannote_pipeline(cfg):
    logging.info("Beginning Pyannote diarization.")
    d = diarization_utils.PyannoteDiarizer(cfg)
    d.diarize()
    create_speaker_segments(os.path.join(cfg.get("audio_folder"), "pyannote-diarization"), filter_sec=cfg.get("rttm/filter_sec"), unified=cfg.get("rttm/unified"))
    diarization_utils.rttm_filter(os.path.join(cfg.get("audio_folder"), "pyannote-diarization"), filter_sec=cfg.get("rttm/filter_sec"))
    logging.info("Pyannote diarization complete.")


def nemo_pipeline(cfg):
    logging.info("Beginning NeMo diarization.")
    d = diarization_utils.NemoDiarizer(cfg)
    d.diarize()
    create_speaker_segments(os.path.join(cfg.get("audio_folder"), "nemo-diarization"), filter_sec=cfg.get("rttm/filter_sec"), unified=cfg.get("rttm/unified"))
    diarization_utils.rttm_filter(os.path.join(cfg.get("audio_folder"), "nemo-diarization"), filter_sec=cfg.get("rttm/filter_sec"))
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

    # copy yaml config to audio folder.
    logging.info("Config information for this diarization run:")
    with open(args.config_path) as cfg_file:
        cfg_dict = yaml.full_load(cfg_file)
        with open(os.path.join(cfg.get('audio_folder', os.getcwd()), 'diarization.log'), 'a') as logfile:
            documents = yaml.dump(cfg_dict, logfile)

    main(cfg)
