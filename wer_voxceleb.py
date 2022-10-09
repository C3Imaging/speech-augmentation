import os
import re
import time
import logging
import argparse
import torchaudio
from tqdm import tqdm
from Tools import decoding_utils


# running average WER for a decoder
class DecoderWER():
    def __init__(self, name):
        self.name = name # the class name of the decoder used
        self.wer_sum = 0.0 # sum of WERs calculated for this decoder so far (instead of having list)
        self.num_wers = 0 # the number of WERs calculated for this decoder so far
        self.avg_wer = 0.0 # the average WER for this decoder so far

    def update_avg_wer(self, wer):
        self.wer_sum += wer
        self.num_wers += 1
        self.avg_wer = self.wer_sum / self.num_wers


def get_wer(ground_truth_transcript_path, decoder_transcript_path):
    with open(ground_truth_transcript_path, 'r') as f:
        ground_truth_transcript = f.readline().strip().lower().split(' ')
    with open(decoder_transcript_path, 'r') as f:
        decoder_transcript = f.readline().strip().lower().split(' ')

    return torchaudio.functional.edit_distance(ground_truth_transcript, decoder_transcript) / len(ground_truth_transcript)


def main():
    # set up the DecoderWER objects, one per decoder from args
    decoders_dict = {}
    for decoder in args.decoders:
        decoders_dict[decoder] = DecoderWER(decoder)
    
    # log the decoders used for this run
    logging.info(f"This run is set up to use the following decoders:")
    for decoder in decoders_dict.keys():
        logging.info(decoder)
    
    # get the number of first level subfolders in voxceleb, for progress bar
    num_folders = len(next(os.walk(args.folder, topdown=True))[1])
    # loop through voxceleb dataset
    for dirpath, _, filenames in tqdm(os.walk(args.folder, topdown=False), total=num_folders, unit=" folders", desc="Calculating avg WER of decoders on dataset, so far"):
        logging.info(f"Starting processing folder: {dirpath}")
        # split all files in directory into a list of speech files and a list of txt files (if they exist,
        # -> transcript files will be contained -> including a ground truth transcript file if it exists)
        _, txt_files = decoding_utils.get_speech_data_lists(dirpath, filenames)
        # initialise dictionary of transcripts paths
        transcripts_paths = {'ground_truths': []}
        for decoder in decoders_dict.keys():
            transcripts_paths[decoder] = []
        # split text files further into ground truth transcripts list and decoder transcripts lists and fill transcripts dictionary
        for f in txt_files:
            if 'ground_truth' in f:
                transcripts_paths['ground_truths'].append(f)
                continue
            else:
                for decoder in decoders_dict.keys():
                    if decoder in f:
                        transcripts_paths[decoder].append(f)
                        break
        # loop through each known decoder
        for decoder in decoders_dict.keys():
            # skip this decoder if there are no transcripts generated by it in the folder
            # avoids accessing empty list
            if len(transcripts_paths[decoder]):
                # loop through each transcript created by this decoder
                for decoder_transcript_path in transcripts_paths[decoder]:
                    ground_truth_transcript_path = ''
                    # check if a corresponding ground truth transcript exists for that decoder transcript by ID (VoxCeleb format)
                    id = re.sub(r'[^0-9]', '', decoder_transcript_path.split('/')[-1])
                    for gt in transcripts_paths['ground_truths']:
                        if id in gt.split('/')[-1]:
                            ground_truth_transcript_path = gt
                            break
                    # calculate WER only for those decoder transcripts that have a corresponding ground truth transcript
                    if len(ground_truth_transcript_path):
                        # update running average WER for that decoder
                        wer = get_wer(ground_truth_transcript_path, decoder_transcript_path)
                        decoders_dict[decoder].update_avg_wer(wer)
                    else:
                        logging.warn(f"No ground truth transcript file was found in the folder {dirpath} for ID {id}.")
            else:
                logging.warn(f"No transcripts were found for {decoder} in the folder {dirpath}.")
        logging.info(f"Finished processing folder: {dirpath}")


if __name__ == "__main__":
    # set up command line arguments
    parser = argparse.ArgumentParser(
        description="Calculate average WER of decoder(s) [specified by --decoders] on the VoxCeleb dataset.")
    parser.add_argument("folder", type=str, nargs='?', default=os.getcwd(),
                        help="Path to VoxCeleb dataset root folder.")
    parser.add_argument("--decoders", nargs="*", type=str, choices=decoding_utils.known_decoders, default=decoding_utils.known_decoders,
                        help="Optional named argument specifying a list of decoder classes to use. By default all implemented decoders are used.")
    # parse command line arguments
    global args
    args = parser.parse_args()

    # setup logging to logfile only
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # track INFO logging events (default was WARNING)
    root_logger.handlers = [] # clear handlers
    root_logger.addHandler(logging.FileHandler(os.path.join(args.folder, 'wer_calculations_run.log'), 'w+')) # handler to log to file
    root_logger.handlers[0].setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s')) # log level and message

     # start timing how long it takes to run script
    tic = time.perf_counter()

    logging.info("Started script.")
    main()
    logging.info("Ended script.")

    toc = time.perf_counter()
    logging.info(f"Finished processing in {time.strftime('%H:%M:%Ss', time.gmtime(toc - tic))}")
