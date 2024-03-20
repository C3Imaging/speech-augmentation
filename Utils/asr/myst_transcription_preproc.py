"""Save a list of audio filepaths to audio files that do not have corresponding transcription files in the original MyST dataset."""


import os
import json
import math
import random
import librosa
import argparse
import collections


def is_two_vals(obj):
    return True if len(obj) == 2 else False


def is_a_percentage(val):
    return True if 0 <= val <= 100 else False


def is_a_percentage_range(obj):
    return True if obj[0] < obj[1] and obj[0] >= 0 and obj[1] <= 100 else False


def only_one_of_two_is_defined(obj1, obj2):
    # return True only if one of two is not None. Return False if both are None or both are not None.
    return True if (obj1 is None) != (obj2 is None) else False


def run_args_checks():
    if args.perc_range: 
        assert is_two_vals(args.perc_range) is True and is_a_percentage_range(args.perc_range) is True, "ERROR: --perc_range must get two percentage values, where the first value is less than the second."
    if args.sampling_perc:
        assert is_a_percentage(args.sampling_perc), "ERROR: --sampling_perc must be a percentage."
    assert only_one_of_two_is_defined(args.perc_range, args.sampling_perc)


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
        
        if args.perc_range:
            # use a slice from the contiguous ordered data list to create the manifest file.
            non_transcribed_wavs.sort()
            l = len(non_transcribed_wavs)
            idx_start = math.floor(l * args.perc_range[0] / 100)
            idx_end = math.floor(l * args.perc_range[1] / 100)
            non_transcribed_wavs = non_transcribed_wavs[idx_start:idx_end]
            
        if args.sampling_perc:
            # use only a random subset of the total filtered untranscribed files, specified as a percentage, to create the manifest file.
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
    parser.add_argument("--perc_range", type=int, nargs='+', required=False,
                        help="Include only a slice of the total data in the manifest file, specified as two int values, the starting percentage and ending percentage of the data. The data is viewed as an ordered contiguous list, i.e. no random sampling. E.g.1: 0 50 means select a slice of the total data from the start to 50 percent, inclusively. E.g.2: 0 100 means use all the data (no need to explicitly use this combination, can just omit this argument). E.g.3: 14 67 means use data from 14 percent to 67 percent, both inclusively.")
    parser.add_argument("--sampling_perc", type=int, required=False,
                        help="Conduct random sampling without replacement to select the specified percentage of the time filtered data as a subset from which to create the manifest file from, if needed. E.g. 100 means 100 percent of data is sampled (no need to explicitly specify this, can just omit this argument).")
    
    # parse command line arguments.
    args = parser.parse_args()
    run_args_checks()
    
    create_manifest()
    
    save_time_stats()
    
    
    
        