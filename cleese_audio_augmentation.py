# -*- coding: utf-8 -*-
"""
Augmenting adult speech samples to sound child-like. Using CLEESE, an open-source python toolbox
for creating transformations on sound:
https://github.com/neuro-team-femto/cleese

"""

import os
import re
import sys
import time
import os.path
import logging
import argparse
import numpy as np
from tqdm import tqdm
from Tools import utils
from Tools import cleese_utils


def shift (input_file, givenBPF, output_dir, transf):
    configFile = '/workspace/projects/Alignment/wav2vec2_alignment/cleeseConfig_all_lj.py'
    transfer = transf
    return cleese_utils.process(soundData=input_file, configFile=configFile, outputDirPath=output_dir, transfer=transfer, BPF=givenBPF)


def apply_pitch_shifting(in_dir, out_dir, female_pitch=0, male_pitch=0):
    transf = 'pitch' #Set the type of transformation (Pitch transformation)
    
    # in_dir must have Librispeech folder structure and the folders inside it must have 'f' or 'm' appended, depending on gender of the speaker.
    # The processing that involves appending gender to folder name is done in the Compute_librispeech_cmukids_similarities.py script prior to running this script.
    # The recording session folders for each speaker folder in in_dir must also have a wav2vec2_alignments/ folder inside them.
    # The processing that involves generating these folders is done in the forced_alignment_librispeech.py script using wav2vec2 ASR model forced alignment prior to running this script.

    # get all the speaker folder names from the processed folder
    spkr_dirs = cleese_utils.list_spkr_dirnames(in_dir)

    # NOTE: example BPF for Cleese process
    # times = [0.1, 0.2]
    # pitch_shifts = [100, 200]
    # bpf = [[0.1, 100], [0.2, 200]]
    # applies pitch shift value 100 from 0.1s until the time specified in the next tuple's time, i.e. until 0.2s, at which point pitch shift of 200 will be applied.

    # loop through each speaker folder
    for spkr_dir in tqdm(spkr_dirs, total=len(spkr_dirs), unit=" speakers", desc="Pitch shifting speakers, so far"):
        # check to ensure speaker folders names have gender suffixes
        if [*spkr_dir][-1].lower() not in ['f', 'm']:
            raise Exception(f'The speaker folder {spkr_dir} does not end with a gender tag! Please ensure all speaker folders have a gender tag by running Compute_librispeech_cmukids_similarities.py script first.')
        if [*spkr_dir][-1].lower() == 'f':
            # NOTE: [0.0, 0.0] must be specified as the first BPF tuple for pitch shifting to work correctly when trying to pitch shift starting at time 0.0
            pitch_givenBPF = np.array([[0.0, 0.0], [0.0, female_pitch]]) #create pitch shift break point function for female voices
            cur_pitch_factor = female_pitch
        else:
            pitch_givenBPF = np.array([[0.0, 0.0], [0.0, male_pitch]])  #create pitch shift break point function for male voices
            cur_pitch_factor = male_pitch
        # loop through the recording sessions folders of a speaker
        recording_sessions = os.listdir(spkr_dir)
        for sess in tqdm(recording_sessions, total=len(recording_sessions), unit=" recording sessions", desc=f"Pitch shifting recording sessions for speaker {spkr_dir.split('/')[-1]}, so far"):
            # Specify new output folder to create, in which to save the augmentations for this speaker.
            # It will be the speaker folder name suffixed with _pitch, while the sessions folders names are left the same.
            # Created by Cleese when calling batched processing with shift() function.
            PITCH_OUTPUTDIR = os.path.join(out_dir, spkr_dir.split('/')[-1] + "_pitch" + str(int(cur_pitch_factor)) + "/" + sess)
            # get all files in the recording session folder
            input_files = cleese_utils.list_file_paths(os.path.join(spkr_dir, sess))
            # get only the audio files
            input_files = list(filter(lambda filepath: filepath.endswith('.flac'), input_files))
            # loop through the files
            for filepath in tqdm(input_files, total=len(input_files), unit=" audio files", desc=f"Pitch shifting audio files in recording session {sess}, so far"):
                pass
                # apply augmentation to each audio file, saving the result to PITCH_OUTPUTDIR
                shift(filepath, pitch_givenBPF, PITCH_OUTPUTDIR, transf)  #Call shift function for single file pitch shifting
            logging.info(f"{PITCH_OUTPUTDIR} created.")
       
       
def apply_time_shifting(in_dir, out_dir, alignments_dir, args, save_transcript=False): 
    transf = 'stretch' #Set the type of transformation (Time stretch transformation)
    # assume time shifting is applied on the Librispeech-format folder that has pitch shifted voices
    # get all the speaker folder names from the root_dir Librispeech-format folder.
    pitch_shifted_dirs = cleese_utils.list_spkr_dirnames(in_dir)
    pitch_shifted_dirs2 = []
    # keep only pitch shift folders that were generated in this run
    for folder in pitch_shifted_dirs:
        gender = '_'.join(folder.split('/')[-1].split('_')[:2]).split('_')[-1]
        # only extract the pitch value PPP from the folders that end in '_pitchPPP'
        matches = re.findall(r'[0-9]+', folder.split('/')[-1].split('_')[-1])
        pitch_lvl = int(matches[0]) if matches else None

        if gender == 'f':
            # if the pitch level found matches the pitch level used in this run
            if pitch_lvl == args.female_pitch:
                pitch_shifted_dirs2.append(folder)
        else:
            # if the pitch level found matches the pitch level used in this run
            if pitch_lvl == args.male_pitch:
                pitch_shifted_dirs2.append(folder)

    # loop through each speaker folder
    for pitch_shifted_dir in tqdm(pitch_shifted_dirs2, total=len(pitch_shifted_dirs2), unit=" speakers", desc="Time stretching speakers, so far"):
        # loop through the recording sessions folders of a speaker
        recording_sessions = os.listdir(pitch_shifted_dir)
        for sess in tqdm(recording_sessions, total=len(recording_sessions), unit=" recording sessions", desc=f"Time stretching recording sessions for speaker {pitch_shifted_dir.split('/')[-1]}, so far"):
            # Specify new output folder to create, in which to save the augmentations, for this speaker's recording session.
            # It will be the speaker folder name suffixed with _stretch (effectively being _pitch_stretch, assuming pitch shifting was the augmentation done prior to time stretching),
            #  while the sessions folders names are left the same.
            STRETCH_OUTPUTDIR = os.path.join(out_dir, pitch_shifted_dir.split('/')[-1] + "_stretch" + "/" + sess)
            # get all files in the recording session folder
            input_files = cleese_utils.list_file_paths(os.path.join(pitch_shifted_dir, sess)) #List input audio files per pitch_shifted directory
            # get only the audio files
            input_files = list(filter(lambda filepath: filepath.endswith('.wav'), input_files))
            # loop through the files
            for filepath in tqdm(input_files, total=len(input_files), unit=" audio files", desc=f"Time stretching audio files in recording session {sess}, so far"):
                # apply augmentation to each audio file, saving the result to STRETCH_OUTPUTDIR
                # for the words as determined by forced alignment
                start_time_list = []
                stop_time_list = []
                # open the forced alignment file for the corresponding audio file
                # the file will contain the start and stop times for each spoken word in the audio file
                original_speaker_id = pitch_shifted_dir.split('/')[-1]
                original_speaker_id = '_'.join(original_speaker_id.split('_')[:2])
                path_to_original_recording_session = os.path.join(alignments_dir, original_speaker_id, sess)
                audio_file_id = filepath.split("/")[-1].split("-")[-1].split(".")[0]
                path_to_audio_alignment = os.path.join("wav2vec2_alignments", audio_file_id, "alignments.txt")
                alignment_filepath = os.path.join(path_to_original_recording_session, path_to_audio_alignment)
                try:
                    with open(alignment_filepath, 'r') as f:
                        for line in f.readlines():                  
                            if "confidence_score" in line:
                                pass
                            else:
                                # append the start time of each word to the start times list, with a slight time padding before the word
                                start_time_list.append(float(line.split(",")[-2]) - 0.002)
                                # append the stop time of each word to the stop times list, with a slight time padding after the word
                                stop_time_list.append(float(line.split(",")[-1].split("\n")[0]) + 0.005)
                    if save_transcript:
                        transcript = utils.get_transcript_from_alignment(alignment_filepath)
                except FileNotFoundError:
                    raise Exception(f'The folder {path_to_original_recording_session} does not have an alignments folder! Please ensure all recording session folders for all speakers have an alignments folder by running forced_alignment_librispeech.py script first.')
                        
                # create time shift BPF function to apply to the audio content
                # make a list of stretch factors for all words
                words_stretch = [1. for _ in range(len(start_time_list))]
                # make a list of stretch factors for all white spaces (silences)
                # the stop times of each word is logically equivalent to the start times of each silence
                space_stretch = [1.8 for _ in range(len(stop_time_list))]
                # Create corresponding timestamps and time shift factors arrays over the entire audio content,
                #  interleaving the starts of words and silences as the edges for the BPF.
                # Each index in time_stamps defines the start time of a word or a silence, and the corresponding index
                #  in stretch_factors defines the time stretch factor to apply starting at that time and ending at the time defined by the next index of time_stamps.
                # These two arrays will later be stacked into corresponding tuples to create the overall BPF for Cleese process to work.
                time_stamps = [*sum(zip(start_time_list, stop_time_list), ())] # combine start time and stop time lists into one, interleaving the list elements.
                stretch_factors = [*sum(zip(words_stretch, space_stretch),())] # combine word stretch factor and space stretch factor lists into one, interleaving the list elements.
                
                time_stamps = np.asarray(time_stamps, dtype=np.float32) #convert time stamps list to array
                stretch_factors = np.asarray(stretch_factors, dtype=np.float32) #convert stretch factors list to array
                
                # creates a list of durations in seconds of each word and silence, chronologically.
                word_space_durations = np.diff(time_stamps) #Get the durations of all words and spaces per audio
                # loop through the durations of the words and silences, chronologically
                for duration_idx in range(len(word_space_durations)):
                    if word_space_durations[duration_idx] > 0.7: #check the value of each duration (looking for longer durations)
                        # assume that word_space_durations[0] is a word and not a silence, meaning words will have an odd index
                        if duration_idx%2 != 0: #and check if longer duration corresponds to a word rather than a space
                            # stretch the long words even more than shorter words, by increasing their corresponding stretch factor from 1.8 to 2.0
                            stretch_factors[duration_idx] = 2.0 #change the stretch factor from 1.8 to 2.0 for longer words 
                # create overall time stretch BPF to apply on the audio content of this wav file
                stretch_givenBPF = np.column_stack((time_stamps, stretch_factors))
                # apply augmentation to the audio file, saving the result to STRETCH_OUTPUTDIR
                naming_format = shift(filepath, stretch_givenBPF, STRETCH_OUTPUTDIR, transf) #Call shift function for single file time stretching
                # save the corresponding transcript in the new folder if required
                if save_transcript:
                    with open(os.path.join(STRETCH_OUTPUTDIR, naming_format+'_tr.txt'), 'w') as f:
                        f.write(transcript.lower())
            logging.info(f"{STRETCH_OUTPUTDIR} created.")


def main(args):
    """The result of this script will be a new folder, whose root name is specified by args.out_dir,
    which will have the Librispeech folder structure, containing recording sessions that have Cleese augmentations
    
    For each speaker from the source folder (specified by args.in_dir), which also has the Librispeech folder structure.
    The args.in_dir folder is first created by running the Compute_librispeech_cmukids_similarities.py script on a Librispeech folder,
     in which the most child-like adult speakers are selected. This will append the gender suffix to the speaker folders.
    The original Librispeech folder remains unchanged.
    Then for that created folder, forced_alignment_librispeech.py is run on it to create the wav2vec2 alignments folders for each recording session.
        
    The speaker folders in args.out_dir will have the augmentation type suffixed to their names.
    """

    apply_pitch_shifting(args.in_dir, args.out_dir, args.female_pitch, args.male_pitch)
    # use the folder created by pitch shifting as the in_dir folder
    # specify the folder in which to find time alignments as alignments_dir
    apply_time_shifting(in_dir=args.out_dir, out_dir=args.out_dir, alignments_dir=args.in_dir, args=args, save_transcript=True) 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run pitch shift and time stretch Cleese augmentations on Librispeech folder.")
    parser.add_argument("--in_dir", type=str, required=True,
                        help="Path to the root of an existing folder that contains speaker folders (in Librispeech folder structure) to process. The audio files within will be left unaltered. The speaker folders must have gender suffixes. The recording sessions folders must include a wav2vec2 alignments folder.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Path of output dir that will be created with augmented speech. Will be in Librispeech folder structure. Can specify the same one for multiple runs. Will have speaker folders suffixed with augmentation type created.")
    parser.add_argument("--female_pitch", type=int, required=True,
                        help="The pitch shift factor expressed in cents to apply to all speech samples of female speakers.")
    parser.add_argument("--male_pitch", type=int, required=True,
                        help="The pitch shift factor expressed in cents to apply to all speech samples of male speakers.")
    # parse command line arguments
    args = parser.parse_args()

    # setup logging to both console and logfile
    utils.setup_logging(args.out_dir, 'cleese_augmentations.log', console=True, filemode='a')

    # start timing how long it takes to run script
    tic = time.perf_counter()
    # log the command that started the script
    logging.info(f"Started script via: {' '.join(sys.argv)}")

    main(args)

    toc = time.perf_counter()
    logging.info(f"Finished processing in {time.strftime('%H:%M:%Ss', time.gmtime(toc - tic))}")
    