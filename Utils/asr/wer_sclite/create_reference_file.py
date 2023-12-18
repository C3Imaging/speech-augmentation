"""
Create a reference.txt file suitable as input for the sclite https://github.com/usnistgov/SCTK/blob/master/doc/sclite.htm
 WER calculation tool using the txt files in an audio dataset directory.
 
The format of the output hypothesis.txt file is the following:

 the hypothesis transcript for audio one (unique id for audio 1)
 the hypothesis transcript for audio two (unique id for audio 2)
 ...
 etc.

Script assumes that for each audiofile there is a corresponding transcript .txt file with the same name, whose content is just the natural language transcript text.
This format for paired audio data is called the LibriTTS format.
"""


import os
import argparse
from pathlib import Path


def main(args):
    # get ground truth transcripts files by searching all text files in all subfolders of the input directory.
    gt_tr_paths = list(map(lambda x: str(x), list(Path(args.in_dir).glob("**/*.txt"))))
    # filter out vocab file, hypothesis.txt file, reference.txt and any alignments.txt files.
    gt_tr_paths = list(filter(lambda x: not ("dict" in x or "hypothesis" in x or "hypotheses" in x or "reference" in x or "alignment" in x), gt_tr_paths))
    # filter out any undesired wav files' corresponding transcripts files.
    if args.path_filters:
        for fil in args.path_filters:
            gt_tr_paths = [t for t in gt_tr_paths if fil not in t]
    gt_tr_paths.sort()

    # if there are ground truth transcripts in the input dataset's folder structure, populate reference.txt in the output directory.
    if len(gt_tr_paths):
        with open(os.path.join(args.out_dir, "reference.txt"), 'w') as ref_file:
            for gt_tr_path in gt_tr_paths:
                # create unique id of audio sample by including leaf folder in the id
                temp = gt_tr_path.split('/')[-2:] # [0] = subfolder, [1] = ____.txt
                temp[-1] = temp[-1].split('.txt')[0] # remove '.txt'
                id = '/'.join(temp)
                f = open(gt_tr_path, "r")
                tr = f.read().strip().lower()
                f.close()
                ref_file.write(f"{tr} ({id})\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="If ground truth transcript files exist, compile and save them in one 'reference.txt' file in a format compatible for input to sclite to calculate WER stats. Script assumes wav files and transcript files have the same names.")
    parser.add_argument("--in_dir", type=str, required=True,
                        help="Path to an existing folder containing paired <audiofile, transcript> data.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Path to an existing output folder where the resultant 'reference.txt' file will be saved.")
    parser.add_argument("--path_filters", type=str, nargs='+', default='',
                        help="List of keywords to filter the paths to wav files/their transcript files in the --in_dir directory. Will filter out any transcript text files that have those keywords present anywhere in their absolute path.")
    
    # parse command line arguments.
    args = parser.parse_args()

    main(args)