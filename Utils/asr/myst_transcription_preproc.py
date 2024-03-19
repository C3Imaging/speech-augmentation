"""Save a list of audio filepaths to audio files that do not have corresponding transcription files in the original MyST dataset."""


import os
import json
import math
import random
import librosa
import argparse
import collections
from datetime import timedelta


def create_manifest():
    """"Save a list of audio filepaths to audio files that do not have corresponding transcription files in the original MyST dataset as a manifest JSON file,
         where each line in the file is a dict in the following format: {"audio_filepath": /path/to/audio.wav, "duration": time in secs}
    """
     
    non_transcribed_wavs = list() # list of non-transcribed audio filepaths.
    for dirpath, subdirs, files in os.walk(args.in_dir, topdown=False):
        temp_wavs = list() # list of non-transcribed wav files in the current subfolder.
        # get all wav files.
        wavs = list(filter(lambda x: x.endswith('.wav'), files))
        if wavs:
            # get all trn files.
            trns = list(filter(lambda x: x.endswith('.trn'), set(files)-set(wavs)))
            if trns:
                # get filenames of all wav files without the extension.
                ids = list(map(lambda x: x.split('.wav')[0], wavs))
                # loop through the wav files and add only those that do not have a corresponding trn file.
                for id in ids:
                    if (id+'.trn') not in trns: temp_wavs.append(os.path.join(dirpath, id+'.wav'))
                # add all non-transcribed wav files to the global list.
                if temp_wavs:
                    non_transcribed_wavs.append(temp_wavs)
            else:
                # if no transcript files present in the subfolder at all, all wav files are untranscribed, so add them all to the global list.
                non_transcribed_wavs.append(list(map(lambda x: os.path.join(dirpath, x), wavs)))

    if non_transcribed_wavs:
        # unpack the list of lists into a single list.
        non_transcribed_wavs = [wavpath for sublist in non_transcribed_wavs for wavpath in sublist]
        
        # filter out audio files that are shorter than the minimum duration allowable by the 'min_time' filter.
        non_transcribed_wavs = list(filter(lambda x: librosa.core.get_duration(filename=x) > args.min_time, non_transcribed_wavs))
        
        if args.sampling_perc < 100:
            # use only a subset of the total filtered untranscribed files, specified as a percentage, to create the manifest file.
            n = math.floor(len(non_transcribed_wavs) * args.sampling_perc / 100)
            non_transcribed_wavs = random.sample(non_transcribed_wavs, n)

        # write the list of non-transcribed audio files' paths to a manifest JSON file.
        # for each wav file, add a line in the following format: {"audio_filepath": /path/to/audio.wav, "duration": time in secs}
        with open(args.out_filepath, 'w') as f:
            for wavpath in non_transcribed_wavs:
                # Write the metadata to the manifest.
                metadata = {
                    "audio_filepath": wavpath,
                    "duration": librosa.core.get_duration(filename=wavpath),
                }
                json.dump(metadata, f)
                f.write('\n')
                
                
def save_time_stats():
    """Load the JSON manifest file and save the dataset's audio duration frequency statistics to another TXT file."""
    times = list() # list of audio times from the JSON manifest file.
    total_time = 0.0 # the total length of time of all the audiofiles.
    with open(args.out_filepath, 'r') as fi:
        for line in fi:
            # get the time duration of the current audio file rounded down to the nearest second.
            t = json.loads(line)['duration']
            total_time += t
            times.append(math.floor(t))
    
    # create time duration frequency table.   
    counter = collections.Counter(times)
    
    # save time duration frequency statistics to file.
    stats_outpath = args.out_filepath.split('.json')[0]+'_timestats.txt'
    with open(stats_outpath, 'w') as fo:
        for t, c in counter.most_common():
            fo.write(f'Count of audios w. dur {t}-{t+1} secs: {c}\n')
        # get min and seconds first
        mm, ss = divmod(total_time, 60)
        # get hours
        hh, mm= divmod(mm, 60)
        fo.write(f"Total time (in HH:MM:SS): {int(hh)}:{int(mm)}:{int(ss)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a list of audio filepaths to audio files that do not have corresponding transcription files in the original MyST dataset.")
    parser.add_argument("--in_dir", type=str, required=True,
                        help="Path to the original MyST dataset root folder.")
    parser.add_argument("--out_filepath", type=str, required=True,
                        help="Full path to a new JSON file to create, that will list the wavpaths of audio files that do not have a corresponding transcript.")
    parser.add_argument("--min_time", type=float, default=0.0,
                        help="Filter to exclude audio files whose duration is less than the specified value.")
    parser.add_argument("--sampling_perc", type=int, default=100,
                        help="Conduct random sampling without replacement to select the specified percentage of the time filtered data as a subset from which to create the manifest file from, if needed. Defaults to entire population, i.e. 100 percent of data used.")
    
    # parse command line arguments.
    args = parser.parse_args()
    
    create_manifest()
    
    save_time_stats()
    
    
    
        