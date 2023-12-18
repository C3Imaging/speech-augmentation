"""
Create a hypothesis.txt file suitable as input for the sclite https://github.com/usnistgov/SCTK/blob/master/doc/sclite.htm
 WER calculation tool using the text predictions found in a JSON output file of an ASR inference script.
 
 The format of the output hypothesis.txt file is the following:

 the hypothesis transcript for audio one (unique id for audio 1)
 the hypothesis transcript for audio two (unique id for audio 2)
 ...
 etc.
"""


import os
import sys
import json
import argparse
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from common_utils import get_audio_id


def main(args):
    with open(args.json_path, 'r', encoding='utf-8') as fr:
        # loop through the JSON file line by line/hypothesis by hypothesis.
        for line in fr:
            item = json.loads(line)
            # do not save to hypothesis.txt the hypotheses for audio files that contain any path filters in their path.
            if args.path_filters:
                if any(fil in item['wav_path'] for fil in args.path_filters):
                    continue
            # write a line to hypothesis.txt for a hypothesis that passes the path filters.
            with open(os.path.join(args.out_dir, "hypothesis.txt"), 'a') as fw:
                fw.write(f"{item['pred_txt']} ({get_audio_id(item['wav_path'])})\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save the hypotheses from an ASR inference JSON output file to a 'hypothesis.txt' file in a format compatible for input to sclite to calculate WER stats.")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to an ASR output file in JSON format containing predicted text hypotheses data.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Path to an existing output folder where the resultant 'hypothesis.txt' file will be saved.")
    parser.add_argument("--path_filters", type=str, nargs='+', default='',
                        help="List of keywords to filter the paths of wav files in the JSON file. Will filter out the hypotheses for audio files that have those keywords present anywhere in their absolute path.")
    
    # parse command line arguments.
    args = parser.parse_args()

    main(args)