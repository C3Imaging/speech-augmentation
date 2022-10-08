import os
import re
import time
import torch
import logging
import argparse
import torchaudio
from tqdm import tqdm
import torchaudio.models.decoder
from Tools import utils
from Tools import decoding_utils


def get_speech_data_lists(dirpath, filenames):
    """Gets the speech audio files paths and the txt files (if they exist) from a single folder.

    Args:
      dirpath (str):
        The path to the directory containing the audio files and some txt files, which the transcript files will be part of (if they exist).
      filenames (list of str elements):
        the list of all files found by os.walk in this directory.

    Returns:
      speech_files (str, list):
        A sorted list of speech file paths found in this directory.
      txt_files (str, list):
        A list of txt files.
    """
    speech_files = []
    txt_files = []
    # loop through all files found
    # split filenames in a directory into speech files and transcript files
    for filename in filenames:
        if filename.endswith('.wav'):
            speech_files.append(os.path.join(dirpath, filename))
        elif filename.endswith('.txt'):
            txt_files.append(os.path.join(dirpath, filename))
    # avoid sorting empty lists
    if len(speech_files):
        speech_files.sort()
    if len(txt_files):
        txt_files.sort()

    return speech_files, txt_files


def infer_and_decode(speechfile, decoder):
    """Runs wav2vec2 inference on a speech file and performs decoding of wav2vec2 output (emissions matrix) by the decoder specified, returning the transcript.
    
    Args:
      speech_file (str):
        Full paths to speech files in a folder.
      decoder (Decoder):
        A Tools.decoding_utils.Decoder object.

    Returns:
      (str):
        Output of the decoder (the decoded transcript)
    """
    with torch.inference_mode():
        # load audio file and resample if needed
        waveform, sample_rate = torchaudio.load(speechfile)
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        # run ASR inference
        emissions, _ = model(waveform.to(device))

        return ' '.join(decoder.forward(emissions)).lower()


def run_inference():
    # create decoders objects and store them in a dictionary using the class name as a key
    # NOTE: only downside to this looping approach is that we can't specify custom args to initialisers, so correct initialisation must be taken care of manually in the class definition
    decoders_dict = {}
    for decoder in args.decoders:
        decoders_dict[decoder] = eval("decoding_utils." + decoder)()

    # log the decoders used for this run
    logging.info(f"This run is set up to use the following decoders:")
    for decoder in decoders_dict.keys():
        logging.info(decoder)

    # get the number of first level subfolders in voxceleb, for progress bar
    num_folders =  len(next(os.walk(args.folder, topdown=True))[1])
    # loop through voxceleb dataset
    for dirpath, _, filenames in tqdm(os.walk(args.folder, topdown=False), total=num_folders, unit=" folders", desc="Transcribing dataset, so far"):
        logging.info(f"Starting processing folder: {dirpath}")
        # split all files in directory into a list of speech files and a list of txt files (if they exist - transcript files will be contained)
        speech_files, txt_files = get_speech_data_lists(dirpath, filenames)
        # get only the transcript files created by decoders
        # create dictionary per decoder where value is a list of unique IDs (voxceleb format) corresponding to speech files for which there already exists a decoded transcript by that decoder in this folder
        decoded_transcript_files_IDs_dict = {}
        for decoder in decoders_dict.keys():
            decoded_transcript_files_IDs_dict[decoder] = []
            # extract only the ID from a decoded transcript filename if the transcript was generated by this decoder
            l = {re.sub(r'[^0-9]', '', f.split('/')[-1]) for f in txt_files if decoder in f}
            # avoid accessing empty list
            if len(l):
                # if there were any decoded transcripts already produced by this decoder in the folder
                decoded_transcript_files_IDs_dict[decoder] = sorted(list(l))
        # loop over speech files in folder
        for speechfile in speech_files:
            # get numerical ID of speech file (voxceleb format)
            id = speechfile.split('/')[-1].split('.wav')[0]
            # loop over decoders
            for d in decoded_transcript_files_IDs_dict.keys():
            # if at least one transcript file exists (avoids accessing empty list)
                if len(decoded_transcript_files_IDs_dict[d]):
                    # skip inference on this audio file if a transcript has already been generated for it by this decoder
                    if utils.BinarySearch(decoded_transcript_files_IDs_dict[d], id):
                        continue
                # if no decoded transcript file already exists for this speech file for this decoder, run inference and decode to generate a transcript with this decoder
                decoded_transcript = infer_and_decode(speechfile, decoders_dict[d])
                # save transcript result to the same folder where the speech file is
                with open(os.path.join(dirpath, f'{id}_{d}_transcript.txt'), 'w') as f:
                    f.write(f"{decoded_transcript}")
        logging.info(f"Finished processing folder: {dirpath}")


def main():
    "Setup and use wav2vec2 model for creating transcripts using different decoding approaches."
    # setup inference model variables
    global bundle, model, labels, dictionary
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H # wav2vec2 model trained for ASR, sample rate = 16kHz
    model = bundle.get_model().to(device) # wav2vec2 model on GPU
    labels = bundle.get_labels() # vocab of chars known to wav2vec2
    dictionary = {c: i for i, c in enumerate(labels)}

    run_inference()


if __name__ == "__main__":
    # currently implemented decoders
    known_decoders = ['GreedyCTCDecoder', 'BeamSearchDecoder']
    # set up command line arguments
    parser = argparse.ArgumentParser(
        description="Run ASR inference on entire VoxCeleb dataset and save transcripts.")
    parser.add_argument("folder", type=str, nargs='?', default=os.getcwd(),
                        help="Path to VoxCeleb dataset root folder.")
    parser.add_argument("--decoders", nargs="*", type=str, choices=known_decoders, default=known_decoders,
                        help="Optional named argument specifying a list of decoder classes to use. By default all implemented decoders are used.")
    # parse command line arguments
    global args
    args = parser.parse_args()

    # setup logging to logfile only
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # track INFO logging events (default was WARNING)
    root_logger.handlers = [] # clear handlers
    root_logger.addHandler(logging.FileHandler(os.path.join(args.folder, 'wav2vec2_alignments_runTEST.log'), 'w+')) # handler to log to file
    root_logger.handlers[0].setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s')) # log level and message

    #setup CUDA config
    torch.random.manual_seed(0)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # start timing how long it takes to run script
    tic = time.perf_counter()

    logging.info("Started script.")
    main()
    logging.info("Ended script.")

    toc = time.perf_counter()
    logging.info(f"Finished processing in {time.strftime('%H:%M:%Ss', time.gmtime(toc - tic))}")