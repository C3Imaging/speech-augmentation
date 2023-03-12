import os
import sys
import argparse
from pydub import AudioSegment
from Tools import librispeech_utils

sys.path.append("/usr/bin/ffmpeg")


def copy_orig_tr():
    """copies over the original .trans.txt files from the source Librispeech dataset's folders to the folders of the target dataset folder."""
    import shutil

    source_dataset = '/workspace/datasets/LibriSpeech-train-clean-100/LibriSpeech/train-clean-100'
    target_dataset = '/workspace/datasets/Librispeech_train_clean_100_top_31_females_Augmented_Samples_for_ASR'
    for dirpath, _, filenames in os.walk(source_dataset, topdown=False): # if topdown=True, read contents of folder before subfolders, otherwise the reverse logic applies
        speaker_id = dirpath.split('/')[-2:-1][0]
        recording_session_id = dirpath.split('/')[-1]
        orig_tr_file = "-".join([speaker_id,recording_session_id]) + '.trans.txt'
        # for speakers in the target directory only
        if os.path.exists(os.path.join(target_dataset,f"{speaker_id}_f_pitch350")):
            # copy original Librispeech transcript file from the source Librispeech folder to the destination folder for the same speaker's recording session's folder
            a = os.path.join(source_dataset,speaker_id,recording_session_id,orig_tr_file)
            b = os.path.join(target_dataset,f"{speaker_id}_f_pitch350",recording_session_id)
            shutil.copy2(a,b)


def main():
    for dirpath, _, filenames in os.walk(args.folder, topdown=asleaf): # if topdown=True, read contents of folder before subfolders, otherwise the reverse logic applies
        # get list of speech files and corresponding transcripts from a single folder
        speech_files, transcripts = librispeech_utils.get_speech_data_lists(dirpath, filenames)
        # process only the leaf folders with the audio files
        if speech_files:
            # convert audio files from .flac to .wav (assume all audio files are the same format, so check only 1st elem)
            if ".flac" in speech_files[0]:
                for speech_file in speech_files:
                    audio = AudioSegment.from_file(speech_file, "flac")
                    audio.export(speech_file.replace("flac", "wav"), format="wav")
                    os.remove(speech_file)
            # process only those folders that contain a Librispeech transcripts text file
            if transcripts is not None:
                # get the list of audio samples' filenames
                ids = list(map(lambda x: x.split('.wav')[0], speech_files))
                # convert the transcripts into lower case sentences
                transcripts = list(map(lambda x: x.replace('|', ' ').lower(), transcripts))
                # write the converted transcripts into individual txt files and save them in the same folder as the audio files
                for id, transcript in zip(ids, transcripts):
                    if ".flac" in id:
                        id = id.replace(".flac", "")
                    with open(os.path.join(dirpath, f'{id}.txt'), 'w') as f:
                        f.write(transcript)
                # remove original Librispeech formatted transcript file from this folder that contains audio files
                if args.rem_orig_tr:
                    orig_tr_filename = list(filter(lambda x: "trans.txt" in x, filenames))[0]
                    if os.path.exists(os.path.join(dirpath,orig_tr_filename)): os.remove(os.path.join(dirpath,orig_tr_filename))
                # remove any "___BPF.txt" files from this folder that contains audio files
                if args.rem_BPF:
                    [os.remove(os.path.join(dirpath,f)) for f in filenames if "BPF.txt" in f]
                # remove "wav2vec2_alignments/" subfolder from this folder
                if args.rem_alignments:
                    import shutil
                    if os.path.exists(os.path.join(dirpath,"wav2vec2_alignments")): shutil.rmtree(os.path.join(dirpath,"wav2vec2_alignments"))
        if asleaf:
            break # to prevent reading subfolders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts format of transcripts of an audio dataset from Librispeech to LibriTTS. Populates folders that contain audio with a transcript file for each audio file.")
    parser.add_argument("folder", type=str, nargs='?', default=os.getcwd(),
                        help="Path to a folder in Librispeech, can be a root folder containing other subfolders, such as speaker subfolders or recording session subfolders, or a leaf folder containing audio and a transcript file. Defaults to CWD if not provided.")
    parser.add_argument("--mode", type=str, choices={'leaf', 'root'}, default="root",
                        help="Specifies how the folder will be processed.\nIf 'leaf': only the folder will be searched for audio files (single folder transcripts conversion),\nIf 'root': subdirs are searched (full dataset transcripts conversion).\nDefaults to 'root' if unspecified.")
    parser.add_argument("--rem_orig_tr", default=False, action='store_true',
                        help="Flag used to specify whether to remove the original transcript file. Defaults to False if flag is not provided.")
    parser.add_argument("--rem_BPF", default=False, action='store_true',
                        help="Flag used to specify whether to remove any '___BPF.txt' files. These are created by CLEESE augmentations. Defaults to False if flag is not provided.")
    parser.add_argument("--rem_alignments", default=False, action='store_true',
                        help="Flag used to specify whether to remove any 'wav2vec2_alignments/' folder from speaker folders, if such exists. These are created by forced alignment script. Defaults to False if flag is not provided.")
    # parse command line arguments
    global args
    args = parser.parse_args()

    # setup directory traversal mode variables
    mode = args.mode
    global asleaf
    asleaf = True if mode == 'leaf' else False

    # copy_orig_tr()

    main()