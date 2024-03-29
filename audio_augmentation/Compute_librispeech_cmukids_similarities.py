# -*- coding: utf-8 -*-
"""
Selecting adult speaker candidates for Adult-to-Child voice conversion. Using the demo01 file from Resemblyzer github repository 
for the computation of cosine similarities between speaker embeddings:
https://github.com/resemble-ai/Resemblyzer/blob/master/demo01_similarity.py

DEMO 01: shows how to compare speech segments (=utterances) between them to get a metric  
on how similar their voices sound. We expect utterances from the same speaker to have a high 
similarity, and those from distinct speakers to have a lower one.
"""

import os
import sys
import logging
import argparse
import warnings
import operator
import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Utils import utils
from pathlib import Path
from itertools import groupby
from distutils.dir_util import copy_tree
from resemblyzer import preprocess_wav, VoiceEncoder

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_adults_wavpaths(path):
     # get a list of all audio files in all subdirs of folder specified by path
    adults_wav_fpaths = list(Path(path).glob("**/*.flac"))
    if not adults_wav_fpaths:
        adults_wav_fpaths = list(Path(path).glob("**/*.wav"))
    assert len(adults_wav_fpaths) != 0, "Adults speech files list is empty. Check the extension used in search."

    return adults_wav_fpaths


def get_kids_wavpaths(path):
    # get a sorted list of all audio files in all subdirs of folder specified by path
    kids_wav_fpaths = list(Path(args.kids_dir).glob("**/*.wav"))
    assert len(kids_wav_fpaths) != 0, "Kids speech files list is empty. Check the extension used in search."
    kids_wav_fpaths.sort()

    return kids_wav_fpaths


def get_genders_dict(path):
    #open text file containing Librispeech/LibriTTS speaker IDs and genders, it is always either called SPEAKERS.txt of SPEAKERS.TXT
    files = ' '.join(os.listdir(path))
    if "SPEAKERS." in files:
        try:
            speakers_info = open(os.path.join(path, "SPEAKERS.TXT"), 'r')
        except FileNotFoundError:
            speakers_info = open(os.path.join(path, "SPEAKERS.txt"), 'r')
    else:
        parent_path = Path(path).parent
        parent_files = os.listdir(parent_path)
        if "SPEAKERS." in parent_files:
            try:
                speakers_info = open(os.path.join(parent_path, "SPEAKERS.TXT"), 'r')
            except FileNotFoundError:
                speakers_info = open(os.path.join(parent_path, "SPEAKERS.txt"), 'r')
    # read info about all speakers from the entire Librispeech collection
    ids = []
    genders = []
    for line in speakers_info.readlines():
        if ";" in line:
            pass
        else:
            l = line.split("|")
            ids.append(l[0].strip())
            genders.append(l[1].strip())

    # create dict of speaker IDs and their gender
    # key=id, value=gender
    ids_and_genders_dict = dict(zip(ids, genders))

    return ids_and_genders_dict


def get_grouped_wavs(adults_wav_fpaths, kids_wav_fpaths, avg_child=True):
    # Group the wavs per speaker and load them into arrays using the preprocessing function provided with 
    #  Resemblyzer to load wavs into memory. This normalizes the volume, trims long silences and resamples 
    #  the wavs to the correct sampling rate. 
    # The grouped, preprocessed audio files will be ready to input to encoder.

    if avg_child:
        # Grouping kids audio samples by each speaker folder's commonly named 'signal' recording session subfolder
        #  effectively combining all child speakers into one folder -> will produce one embedding representing the average CMU child.
        kids_speaker_wavs = {speaker: list(map(preprocess_wav, wav_fpaths)) for speaker, wav_fpaths in
                        groupby(tqdm(kids_wav_fpaths, "Preprocessing kids wavs", len(kids_wav_fpaths), unit="wavs"), 
                                lambda wav_fpath: wav_fpath.parent.stem)}
    else:
        # grouping kids audio samples by speaker ID -> will produce an average embedding for each child speaker
        kids_speaker_wavs = {speaker: list(map(preprocess_wav, wav_fpaths)) for speaker, wav_fpaths in
                        groupby(tqdm(kids_wav_fpaths, "Preprocessing kids wavs", len(kids_wav_fpaths), unit="wavs"), 
                                lambda wav_fpath: wav_fpath.parent.parent.stem)}

    # grouping adults audio samples by speaker ID -> will produce an average embedding for each adult speaker
    adults_speaker_wavs = {speaker: list(map(preprocess_wav, wav_fpaths)) for speaker, wav_fpaths in
                    groupby(tqdm(adults_wav_fpaths, "Preprocessing adults wavs", len(adults_wav_fpaths), unit="wavs"), 
                            lambda wav_fpath: wav_fpath.parent.parent.stem)}

    return adults_speaker_wavs, kids_speaker_wavs


def get_embeds(adults_speaker_wavs, kids_speaker_wavs):
    # initialise the speaker encoder network from the original d-vectors paper https://arxiv.org/abs/1710.10467
    encoder = VoiceEncoder()

    # for each speaker, create an averaged speaker embedding vector of len 256
    #  by averaging all the individual embeddings generated by speaker encoder for each audio sample of that speaker.
    # applies L2 norm to averages after.
    # each embeddings matrix is of shape (num_speakers, embed_size).
    # one matrix for the kids speakers and one for the adults.
    embeds_adults = np.array([encoder.embed_speaker(wavs[:len(wavs)]) for wavs in tqdm(adults_speaker_wavs.values(), total=len(adults_speaker_wavs.values()), unit=" adult speaker", desc=f"Calculating speaker embeddings for adult speakers")])
    embeds_kids = np.array([encoder.embed_speaker(wavs[:len(wavs)]) for wavs in tqdm(kids_speaker_wavs.values(), total=len(kids_speaker_wavs.values()), unit=" child speaker", desc=f"Calculating speaker embeddings for child speakers")])

    return embeds_adults, embeds_kids


def compute_similarities(adults_speaker_wavs, kids_speaker_wavs, embeds_adults, embeds_kids, ids_and_genders_dict):
    # Initialise a dict of 'speaker ID: similarity score' pairs for adult speakers that pass the similarity score threshold,
    #  used for creating speaker_similarities.txt
    adults_high_similarity = {}

    # Compute the similarity matrix. The similarity of two embeddings is simply their dot 
    # product, because the similarity metric is the cosine similarity and the embeddings are 
    # already L2-normed.

    # Short version:
    #utt_sim_matrix = np.inner(embeds_a, embeds_b)

    # Long, detailed version:
    # this: utt_sim_matrix2[i,j] -> will store similarity between adult speaker i and child speaker j
    utt_sim_matrix = np.zeros((len(embeds_adults), len(embeds_kids)))
    # loop through each adult average embedding
    for i in range(len(embeds_adults)):
        adult_speaker_id = list(adults_speaker_wavs.keys())[i]
        logging.info(f"---- current adult speaker ID: {adult_speaker_id} ----")
        # loop through each child average embedding
        for j in range(len(embeds_kids)):
            child_speaker_id = list(kids_speaker_wavs.keys())[j]
            # The @ notation is exactly equivalent to np.dot(embeds_a[i], embeds_b[i])
            utt_sim_matrix[i,j] = embeds_kids[j] @ embeds_adults[i]
            logging.info(f"Similarity score with child speaker ID: '{'avg_CMU_child' if child_speaker_id == 'signal' else child_speaker_id}' ----> {str(utt_sim_matrix[i,j])}")
            # if similarity between the adult speaker and child speaker is considered high
            if (utt_sim_matrix[i,j] > float(args.sim_thresh)):
                logging.info(f"High similarity! (>{args.sim_thresh})")
                adults_high_similarity[adult_speaker_id] = utt_sim_matrix[i,j]
                # get path to adult speaker folder
                adult_path = os.path.join(args.adults_dir, adult_speaker_id)
                # copy only female
                # if ids_and_genders_dict[adult_speaker_id] == 'F':
                # copy everything inside the adult speaker folder (which contains recording sessions subfolders) into OUTPUTDIR/adult_speaker_id
                copy_tree(adult_path, os.path.join(args.out_dir, adult_speaker_id))
                logging.info(f"{os.path.join(args.out_dir, adult_speaker_id)} folder created by copying {adult_path}. Will be suffixed with gender later.")

    return utt_sim_matrix, adults_high_similarity


def main(args):
    adults_wav_fpaths = get_adults_wavpaths(args.adults_dir)
    kids_wav_fpaths = get_kids_wavpaths(args.kids_dir)

    ids_and_genders_dict = get_genders_dict(args.adults_dir)

    # specify if grouping all child speakers into one 'CMU speaker' by setting avg_child to True
    adults_speaker_wavs, kids_speaker_wavs = get_grouped_wavs(adults_wav_fpaths, kids_wav_fpaths, avg_child=True)

    embeds_adults, embeds_kids = get_embeds(adults_speaker_wavs, kids_speaker_wavs)

    utt_sim_mx, adults_high_sim = compute_similarities(adults_speaker_wavs, kids_speaker_wavs, embeds_adults, embeds_kids, ids_and_genders_dict)

    # sort similarities in descending order, using sorted() for backwards compatibility (below Python v3.7), returns a sorted list of tuples.
    adults_high_sim_sorted = sorted(adults_high_sim.items(), key=operator.itemgetter(1), reverse=True)

    with open(os.path.join(args.out_dir, 'speaker_similarities.txt'), "w") as f:
        f.write(f"Similarity scores of adult speakers from {args.adults_dir} that are above {args.sim_thresh}.\n")
        f.write(f"Total number of such speakers: {len(adults_high_sim_sorted)} out of {len(adults_speaker_wavs)}.\n")
        f.write(f"Speaker ID, Gender, Similarity score\n")
        for id, sim in adults_high_sim_sorted:
            f.write(f"{id},{ids_and_genders_dict[id]},{sim}\n")

    #Rename the output adult speaker folders (adding gender '_m' or '_f' to the speaker id folder name)
    speaker_folders = next(os.walk(args.out_dir))[1] # get the speaker folders names we copied
    for speaker_folder in speaker_folders:
        dir_path = os.path.join(args.out_dir, speaker_folder)
        new_name = dir_path + "_" + ids_and_genders_dict[speaker_folder].lower()
        os.rename(dir_path, new_name)
        logging.info(f"{dir_path} renamed to {new_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run speaker encoder on a Librispeech/LibriTTS-formatted dataset and create new folder with speakers that have highest cosine similarity in embeddings space compared to average CMU kids speaker.")
    parser.add_argument("--adults_dir", type=str, required=True,
                        help="Path to an existing folder (has Librispeech/LibriTTS folder structure) containing adult speakers folders. Example: /path/to/LibriSpeech-train-clean-100/LibriSpeech/train-clean-100")
    parser.add_argument("--kids_dir", type=str, required=True,
                        help="Path to an existing folder (has CMU folder structure) containing child speakers folders.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Path to a new output folder (will have Librispeech/LibriTTS folder structure) to create that will contain adult speakers with similarity scores above threshold. Speaker folder names will be suffixed with gender.")
    parser.add_argument("--sim_thresh", type=str, required=True,
                        help="The cosine similarity threshold between an adult speaker and child speaker to filter adult speakers by.")

    # parse command line arguments
    args = parser.parse_args()

    # setup logging to both console and logfile.
    utils.setup_logging(args.out_dir, 'adult_speakers_filtering.log', console=True, filemode='a')

    p = utils.Profiler()
    p.start()

    main(args)

    p.stop()
