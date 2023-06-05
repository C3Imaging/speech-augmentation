import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, current + '/..')
import logging
import argparse
from Tools import utils
import pyannote_diarization_utils
import resemblyzer_diarization_utils
import nemo_diarization_utils


def pyannote_pipeline(root_path, num_speakers=None, resemblyzer_preprocessing=False, filter_sec=0.0, rem_files=False):
    logging.info("Beginning Pyannote diarization.")
    pyannote_diarization_utils.pyannote_diarization(root_path, num_speakers=num_speakers, resemblyzer_preprocessing=resemblyzer_preprocessing)
    # 'pyannote-diarization/' subfolder contains subfolders for each audio recording, each in turn containing a 'diarization.rttm' file.
    # The parent folder of 'pyannote-diarization/' (i.e. root_path) should have the audio recordings wav files.
    for dirpath, subdirs, _ in os.walk(os.path.join(root_path, "pyannote-diarization"), topdown=True):
        for subdir in subdirs:
            # loop only over the subfolders that represent audio files.
            rttm = [f for f in os.listdir(os.path.join(dirpath, subdir)) if f.endswith("rttm")]
            if rttm:
                assert len(rttm) == 1, f"ERROR: there should only be one RTTM file in {os.path.join(dirpath, subdir)}!!!"
                wav_path = os.path.join(dirpath, "resemblyzer_preproc_audio", subdir + ".wav") if resemblyzer_preprocessing else os.path.join("/".join(dirpath.split("/")[:-1]), subdir + ".wav")
                pyannote_diarization_utils.rttm_to_wav(os.path.join(dirpath, subdir, rttm[0]), wav_path, filter_sec=filter_sec, rem_files=rem_files)
        break # loop only over the first level of subdirs (where the audio file folders reside).
    logging.info("Pyannote diarization complete.")


def resemblyzer_pipeline(root_path, similarity_threshold, global_speaker_embeds):
    logging.info("Beginning Resemblyzer diarization.")
    resemblyzer_diarization_utils.resemblyzer_diarization(root_path, similarity_threshold=similarity_threshold, global_speaker_embeds=global_speaker_embeds)
    logging.info("Resemblyzer diarization complete.")


def nemo_pipeline(root_path, num_speakers=None, filter_sec=0.0, rem_files=False):
    logging.info("Beginning NeMo diarization.")
    nd = nemo_diarization_utils.NemoDiarizer(root_path)
    nd.nemo_diarization(num_speakers=num_speakers, filter_sec=filter_sec, rem_files=rem_files)
    logging.info("NeMo diarization complete.")


def main(args):
    utils.mp3_to_wav(args.folder)
    utils.audio_preprocessing(args.folder, in_place=True)

    diarizers = args.diarizers
    if type(diarizers) != list:
        diarizers = [diarizers]
    
    if "pyannote" in diarizers:
        pyannote_pipeline(args.folder, args.num_speakers, resemblyzer_preprocessing=args.resemblyzer_preprocessing, filter_sec=args.filter_sec, rem_files=args.rem_files)
    if "resemblyzer" in diarizers:
        resemblyzer_pipeline(args.folder, args.similarity_threshold, args.global_speaker_embeds)
    if "nemo" in diarizers:
        nemo_pipeline(args.folder, args.num_speakers, filter_sec=args.filter_sec, rem_files=args.rem_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Splits audio files in a folder into speaker-specific diarized audio snippets.")
    parser.add_argument("folder", type=str, nargs='?', default=os.getcwd(),
                        help="Path to a folder containing audio files. Defaults to CWD if not provided.")
    parser.add_argument("--diarizers", type=str, choices=["pyannote", "resemblyzer", "nemo"], nargs='+', default="pyannote",
                        help="Specifies which diarizer model to use.\nIf 'pyannote': only the Pyannote model will be used, etc.,\nIf '[pyannote,resemblyzer]' or vice versa: both will be used.\nDefaults to 'pyannote' if flag is given but command line arg is unspecified or if the flag is not provided at all.")
    parser.add_argument("--num_speakers", type=int, nargs='?', default=None,
                        help="Specifies the number of speakers, if known in advance. Used explicitly by Pyannote and NeMo, otherwise number of speakers will be determined automatically.")
    parser.add_argument("--resemblyzer_preprocessing", default=False, action='store_true',
                        help="Flag used to specify whether to preprocess audio files in Resemblyzer style when using the Pyannote diarizer. Defaults to False if flag is not provided.")
    parser.add_argument("--similarity_threshold", type=float, default=0.7,
                        help="Specifies the speaker embedding similarity confidence threshold for Resemblyzer to use. Defaults to 0.7 if not provided.")
    parser.add_argument("--global_speaker_embeds", default=False, action='store_true',
                        help="Flag used to specify if Resemblyzer will be using the same speaker embeddings for all wav files. Defaults to False if flag is not specified.")
    parser.add_argument("--filter_sec", type=float, default=0.0,
                        help="Used to exclude in the resultant unified speaker audio file those speaker audio segments as defined in the RTTM files that are less than 'filter_sec' seconds in length. Used by Pyannote and NeMo diarizers.")
    parser.add_argument("--rem_files", default=False, action='store_true',
                        help="Flag used to specify whether to remove the intermediate speaker segments audio files, keeping only the resultant unified speaker audio file. Used by Pyannote and NeMo diarizers. Defaults to False if flag is not provided.")
    # parse command line arguments
    args = parser.parse_args()

    # setup logging to both console and logfile
    utils.setup_logging(args.folder, 'diarization.log', console=True, filemode='a')
    # log the command that started the script
    logging.info(f"Started script via: python {' '.join(sys.argv)}")
    
    main(args)
