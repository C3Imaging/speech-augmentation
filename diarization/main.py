import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, current + '/..')
import logging
import argparse
from Tools import utils
import pyannote_diarization_utils
import resemblyzer_diarization_utils


def pyannote_pipeline(root_path, num_speakers, resemblyzer_preprocessing=False):
    logging.info("Beginning Pyannote diarization.")
    utils.mp3_to_wav(root_path)
    utils.audio_preprocessing(root_path, in_place=True)
    pyannote_diarization_utils.pyannote_diarization(root_path, num_speakers=num_speakers, resemblyzer_preprocessing=resemblyzer_preprocessing)
    pyannote_diarization_utils.rttm_to_wav(os.path.join(root_path, "pyannote-diarization"), resemblyzer_preprocessing=resemblyzer_preprocessing)
    logging.info("Pyannote diarization complete.")


def resemblyzer_pipeline(root_path):
    logging.info("Beginning Resemblyzer diarization.")
    resemblyzer_diarization_utils.resemblyzer_diarization(root_path, similarity_threshold=0.7)
    logging.info("Resemblyzer diarization complete.")


def main():
    global root_path, diarizers, num_speakers, resemblyzer_preprocessing
    if "pyannote" in diarizers:
        pyannote_pipeline(root_path, num_speakers, resemblyzer_preprocessing=resemblyzer_preprocessing)
    if "resemblyzer" in diarizers:
        resemblyzer_pipeline(root_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Splits audio files in a folder into speaker-specific diarized audio snippets.")
    parser.add_argument("folder", type=str, nargs='?', default=os.getcwd(),
                        help="Path to a folder containing audio files. Defaults to CWD if not provided.")
    parser.add_argument("--diarizers", type=str, choices=["pyannote", "resemblyzer"], nargs='+', default="pyannote",
                        help="Specifies which diarizer model to use.\nIf 'pyannote': only the Pyannote model will be used,\nIf 'resemblyzer': only the Resemblyzer model will be used,\nIf '[pyannote,resemblyzer]' or vice versa: both will be used.\nDefaults to 'pyannote' if flag is given but command line arg is unspecified or if the flag is not provided at all.")
    parser.add_argument("--num_speakers", type=int, nargs='?', default=0,
                        help="Specifies the number of speakers, if known in advance. Used explicitly by Pyannote, otherwise number of speakers will be determined automatically.")
    parser.add_argument("--resemblyzer_preprocessing", default=False, action='store_true',
                        help="Flag used to specify whether to preprocess audio files in Resemblyzer style when using the Pyannote diarizer. Defaults to False if flag is not provided.")
    # parse command line arguments
    global args
    args = parser.parse_args()
    root_path = args.folder
    diarizers = args.diarizers
    if type(diarizers) != list:
        diarizers = [diarizers]
    num_speakers = args.num_speakers
    resemblyzer_preprocessing = args.resemblyzer_preprocessing
    # root_path = "/workspace/datasets/Wearable_Audio_test"


    # setup logging to both console and logfile
    utils.setup_logging(root_path, 'diarization.log', console=True, filemode='a')
    main()
