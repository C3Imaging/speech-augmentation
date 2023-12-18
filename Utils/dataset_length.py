import os
import argparse
from tqdm import tqdm
import scipy.io.wavfile as wav


def audio_file_path_names(dirname):
    """Return a list of all wav or flac files in all subdirectories of a folder."""
    file_paths = []  
    for root, dirs, files in os.walk(dirname):
        for filename in files:
            filepath = os.path.join(root, filename)
            if filepath.endswith(".wav") or filepath.endswith(".flac"):
                file_paths.append(filepath)
    return file_paths


def countdir(DIRNAME):
    dt = {'0-5':0,'5-10':0,'10-20':0,'20-30':0,'30-40':0,'40-50':0,'50-60':0,'60-70':0,'70-80':0,'80-90':0,'90-100':0,'above 100': 0}
    x=y=y1=y2=y3=y4=y5=y6=y7=y8=y9=y10=y11=0
    audio_files = audio_file_path_names(DIRNAME) # get all file-paths of all audio files in dirname and subdirectories
    for file in tqdm(audio_files): # execute for each file
        (source_rate, source_sig) = wav.read(file)
        duration_seconds = len(source_sig) / float(source_rate) # duration of audio file
        if (duration_seconds > 0) and (duration_seconds <=5) : 
            dt['0-5'] = dt['0-5'] + 1 
            y = y + duration_seconds
        elif (duration_seconds > 5) and (duration_seconds <=10) : 
            dt['5-10'] = dt['5-10'] + 1
            y1 = y1 + duration_seconds
        elif (duration_seconds > 10) and (duration_seconds <=20) : 
            dt['10-20'] = dt['10-20'] + 1
            y2 = y2 + duration_seconds
        elif (duration_seconds > 20) and (duration_seconds <=30) :
            dt['20-30'] = dt['20-30'] + 1
            y3 = y3 + duration_seconds
        elif (duration_seconds > 30) and (duration_seconds <=40) : 
            dt['30-40'] = dt['30-40'] + 1
            y4 = y4 + duration_seconds
        elif (duration_seconds > 40) and (duration_seconds <=50) : 
            dt['40-50'] = dt['40-50'] + 1
            y5 = y5 + duration_seconds
        elif (duration_seconds > 50) and (duration_seconds <=60) : 
            dt['50-60'] = dt['50-60'] + 1
            y6 = y6 + duration_seconds
        elif (duration_seconds > 60) and (duration_seconds <=70) : 
            dt['60-70'] = dt['60-70'] + 1
            y7 = y7 + duration_seconds
        elif (duration_seconds > 70) and (duration_seconds <=80) : 
            dt['70-80'] = dt['70-80'] + 1
            y8 = y8 + duration_seconds
        elif (duration_seconds > 80) and (duration_seconds <=90) : 
            dt['80-90'] = dt['80-90'] + 1
            y9 = y9 + duration_seconds
        elif (duration_seconds > 90) and (duration_seconds <=100) : 
            dt['90-100'] = dt['90-100'] + 1
            y10 = y10 + duration_seconds
        elif (duration_seconds > 100) : 
            dt['above 100'] = dt['above 100']+1 
            y11 = y11 + duration_seconds
        x = x+duration_seconds
    
    print('\nProcessed directory: ', args.folder, '\n')
    print('Total duration ==', x/60, 'mins ==', x/60/60, 'hrs\n')
    print('Total number of audio files: ', len(audio_files), '\n')
    print('Count in seconds for each range: \n', dt)
    print('Duration 0-5s ==', y/60, 'mins ==', y/60/60, 'hrs \n')
    print('Duration 5-10s ==', y1/60, 'mins ==', y1/60/60, 'hrs\n')
    print('Duration 10-20s ==', y2/60, 'mins ==', y2/60/60, 'hrs\n')
    print('Duration 20-30s ==', y3/60, 'mins ==', y3/60/60, 'hrs\n')
    print('Duration 30-40s ==', y4/60, 'mins ==', y4/60/60, 'hrs\n')
    print('Duration 40-50s ==', y5/60, 'mins ==', y5/60/60, 'hrs\n')
    print('Duration 50-60s ==', y6/60, 'mins ==', y6/60/60, 'hrs\n')
    print('Duration 60-70s ==', y7/60, 'mins ==', y7/60/60, 'hrs\n')
    print('Duration 70-80s ==', y8/60, 'mins ==', y8/60/60, 'hrs\n')
    print('Duration 80-90s ==', y9/60, 'mins ==', y9/60/60, 'hrs\n')
    print('Duration 90-100s ==', y10/60, 'mins ==', y10/60/60, 'in hrs\n')
    print('Duration above 100s ==', y11/60, 'mins ==', y11/60/60, 'hrs\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate duration statistics of an audio dataset.")
    parser.add_argument("folder", type=str, nargs='?', default=os.getcwd(),
                        help="Path to a dataset folder. Defaults to CWD if not provided.")
    args = parser.parse_args()
    countdir(args.folder)