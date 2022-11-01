# -*- coding: utf-8 -*-
"""
Augmenting adult speech samples to sound child-like. Using CLEESE, an open-source python toolbox
for creating transformations on sound:
https://github.com/neuro-team-femto/cleese

"""


import os
import os.path

from cleese.cleese import cleeseProcess
import numpy as np
import shutil
import time
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

# existing folder (in Librispeech format) to process, the audio files within will be left unaltered, must include wav2vec2 alignments
DIRSPEAKERS = r'/workspace/projects/Alignment/wav2vec2_alignment/speaker_encoder_outputs/Librispeech-similarity-above-0.75'
# output dir that will be in Librispeech format which will contain augmented audio, can be used for multiple runs
OUTPUTDIR_BASE = r'/workspace/projects/Alignment/wav2vec2_alignment/cleese_outputs/Librispeech-similarity-above-0.75_augment1'


# example BPF for Cleese process
# times = [0.1, 0.2]
# pitch_shifts = [100, 200]
# bpf = [[0.1, 100], [0.2, 200]]
# applies pitch shift value 100 from 0.1s until the time specified in the next tuple's time, i.e. until 0.2s, at which point pitch shift of 200 will be applied.
FEMALE_PITCH_SHIFT_FACTOR = 400.0
MALE_PITCH_SHIFT_FACTOR = 700.0

def list_spkr_dirnames(dir_of_spkrs):
    spkr_dir_list = []
    for spkr_id in os.listdir(dir_of_spkrs):
        print(spkr_id)
        if os.path.isdir(os.path.join(dir_of_spkrs, spkr_id)):
            spkr_dir_list.append(os.path.join(dir_of_spkrs, spkr_id))
    
    return spkr_dir_list
        

def list_file_paths (dirname):
    filepathlist = []
    
    for root, directories, files in os.walk(dirname):
        for filename in (files):
            filepath = os.path.join(root, filename)
            filepathlist.append(filepath)
        return filepathlist
            

def shift (input_file, givenBPF, output_dir, transf):
    configFile = '/workspace/projects/Alignment/wav2vec2_alignment/cleeseConfig_all_lj.py'
    transfer = transf
    cleeseProcess.process(soundData=input_file, configFile=configFile, outputDirPath=output_dir, transfer=transfer, BPF=givenBPF)
    
def apply_pitch_shifting(root_dir):
    # this function works with Librispeech dataset format of root_dir.

    transf = 'pitch' #Set the type of transformation (Pitch transformation)
    
    # Folders inside the root_dir Librispeech-format folder must have 'f' or 'm' appended, depending on gender of the speaker.
    # The processing that involves appending gender to folder name is done in the Compute_librispeech_cmukids_similarities.py script prior to running this script.
    # The recording session folders for each speaker folder in root_dir folder must also have a wav2vec2_alignments/ folder inside them.
    # The processing that involves generating these folders is done in the forced_alignment_librispeech.py script using wav2vec2 ASR model forced alignment prior to running this script.

    # get all the speaker folder names from the processed root_dir Librispeech folder.
    spkr_dirs = list_spkr_dirnames(root_dir)
    # loop through each speaker folder
    for spkr_dir in spkr_dirs:
        if spkr_dir.split()[-1] == 'f':
            # NOTE: [0.0, 0.0] must be specified as the first BPF tuple for pitch shifting to work correctly when trying to pitch shift starting at time 0.0
            pitch_givenBPF = np.array([[0.0, 0.0], [0.0, FEMALE_PITCH_SHIFT_FACTOR]]) #create pitch shift break point function for female voices
            cur_pitch_factor = FEMALE_PITCH_SHIFT_FACTOR
        else:
            pitch_givenBPF = np.array([[0.0, 0.0], [0.0, MALE_PITCH_SHIFT_FACTOR]])  #create pitch shift break point function for male voices
            cur_pitch_factor = MALE_PITCH_SHIFT_FACTOR
        # loop through the recording sessions folders of a speaker
        for subdir in os.listdir(spkr_dir):
            # Specify new output folder to create, in which to save the augmentations for this speaker.
            # Will basically create a speaker folder but suffixed with _pitch, keeping the same recording session subfolders names.
            # Created by Cleese when calling batched processing with shift() function.
            PITCH_OUTPUTDIR = os.path.join(OUTPUTDIR_BASE, spkr_dir.split('/')[-1] + "_pitch" + str(cur_pitch_factor) + "/" + subdir)
            # get all files in the recording session folder
            input_files = list_file_paths(os.path.join(spkr_dir, subdir))
            # loop through the files
            for filepath in input_files:
                if filepath.endswith('.flac'):
                    pass
                    # apply augmentation to each audio file, saving the result to PITCH_OUTPUTDIR
                    shift(filepath, pitch_givenBPF, PITCH_OUTPUTDIR, transf)  #Call shift function for single file pitch shifting
       
       
def apply_time_shifting(root_dir): 
    # this function works with Librispeech dataset format of root_dir.

    transf = 'stretch' #Set the type of transformation (Time stretch transformation)
    # assume time shifting is applied on the Librispeech-format folder that has pitch shifted voices
    # get all the speaker folder names from the root_dir Librispeech-format folder.
    pitch_shifted_dirs = list_spkr_dirnames(root_dir)
    # loop through each speaker folder
    for pitch_shifted_dir in pitch_shifted_dirs:
        # loop through the recording sessions folders of a speaker
        for subdir in os.listdir(pitch_shifted_dir):
            # specify new output folder to create, in which to save the augmentations, for this speaker's recording session.
            # it will be the recording session folder name suffixed with _stretch (effectively being _pitch_stretch, assuming pitch shifting was the augmentation done prior to time shifting)
            STRETCH_OUTPUTDIR = os.path.join(OUTPUTDIR_BASE, pitch_shifted_dir.split('/')[-1] + "_stretch" + "/" + subdir)
            # get all files in the recording session folder
            input_files = list_file_paths(os.path.join(pitch_shifted_dir, subdir)) #List input audio files per pitch_shifted directory
            # loop through the files
            for filepath in input_files:
                if filepath.endswith('.wav'):
                    # apply augmentation to each audio file, saving the result to STRETCH_OUTPUTDIR
                    # for the words as determined by forced alignment
                    start_time_list = []
                    stop_time_list = []
                    # open the forced alignment file for the corresponding audio file
                    # the file will contain the start and stop times for each spoken word in the audio file
                    original_speaker_id = pitch_shifted_dir.split('/')[-1]
                    original_speaker_id = '_'.join(original_speaker_id.split('_')[:-1])
                    path_to_original_recording_session = os.path.join(DIRSPEAKERS, original_speaker_id, subdir)
                    audio_file_id = filepath.split("/")[-1].split("-")[-1].split(".")[0]
                    path_to_audio_alignment = os.path.join("wav2vec2_alignments", audio_file_id, "alignments.txt")
                    # path_to_audio_alignment = "/wav2vec2_alignments/" + str(filepath.split("/")[-1].split("-")[-1].split(".")[0]) + "/alignments.txt" # alignment text file for this audio file
                    alignment_filepath = os.path.join(path_to_original_recording_session, path_to_audio_alignment)
                    with open(alignment_filepath, 'r') as f:
                        for line in f.readlines():                  
                            if "confidence_score" in line:
                                pass
                            else:
                                # append the start time of each word to the start times list, with a slight time padding before the word
                                start_time_list.append(float(line.split(",")[-2]) - 0.002)
                                # append the stop time of each word to the stop times list, with a slight time padding after the word
                                stop_time_list.append(float(line.split(",")[-1].split("\n")[0]) + 0.005)
                            
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
                    shift(filepath, stretch_givenBPF, STRETCH_OUTPUTDIR, transf) #Call shift function for single file time stretching
                   
def main():
    #Set values to compute total runtime 
    start_time  = time.perf_counter()
    logger.debug("Processing started...")
    print("Processing started...")
 
    # the result of this script will be a new folder, whose root name is specified by OUTPUTDIR_BASE,
    # which will be in Librispeech format, containing wav2vec2 alignments and Cleese augmentations folders
    # in each recording session folder for each speaker from source DIRSPEAKERS folder (also in Librispeech format).
    # the source DIRSPEAKERS folder is typically one created by the Compute_librispeech_cmukids_similarities.py script,
    #  in which the most child-like adult speakers are selected.
    # Then for that dataset, forced_alignment_librispeech.py is run on it to create the wav2vec2 alignments folders.
    
    """Call batch pitch shifting function (In the root directory, each directory containing audio files must also 
    contain an alignments directory with corresponding alignments )"""
    # use a folder in Librispeech format, specified by DIRSPEAKERS, which was generated by the Compute_librispeech_cmukids_similarities.py script,
    # which appended 'f' or 'm' to the speakers' folder names, depending on their gender.
    # The voices for each speaker in this folder remain unchanged and are the original recordings from Librispeech. 
    # NOTE: There must be another folder in Librispeech format that contains a wav2vec2_alignments/ folder for each recording session folder for each speaker.
    # To get these folders, run the forced_alignment_librispeech.py script on the folder generated by the Compute_librispeech_cmukids_similarities.py script first.
    apply_pitch_shifting(DIRSPEAKERS) 
    
    """Call batch time stretching funciton"""
    # use the folder created by pitch shifting as the root folder (Librispeech format)
    apply_time_shifting(OUTPUTDIR_BASE) 
    
    print("Processing ended.")
    
    end_time = time.perf_counter()
    print(f"Finished in {time.strftime('%H:%M:%Ss', time.gmtime(end_time - start_time))}")
    

if __name__ == "__main__":
    
    main()
    