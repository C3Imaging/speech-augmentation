"""
Common functions used across different ASR scripts.
Classes and helper functions to create common interface for further processing the outputs of different ASR models that operate under different frameworks.
"""


from dataclasses import dataclass, field, asdict
from typing import List, Any, Union, Optional, Tuple, Dict
import json
import os
from pathlib import Path


def get_wavlist_from_folder(in_dir, path_filters):
    """Get list of wav files across all subfolders and filter out any undesired wav files by name."""

    wav_paths = list(map(lambda x: str(x), list(Path(in_dir).glob("**/*.wav"))))
    # filter out any undesired wav files.
    if path_filters:
        for fil in path_filters:
            wav_paths = [w for w in wav_paths if fil not in w]
    wav_paths.sort()

    return wav_paths


def get_wavlist_from_manifest(filepath, path_filters):
    """
    Get list of wav files from a manifest JSON file and filter out any undesired wav files by name.
    Each line in the JSON file should be a dict containing a 'audio_filepath' field.
    """
    wavpaths = list()
    with open(filepath, mode="r") as f:
        for line in f:
            item = json.loads(line)
            wavpaths.append(item['audio_filepath'])
    # filter out any undesired wav files.
    if path_filters:
        for fil in path_filters:
            wavpaths = [w for w in wavpaths if fil not in w]
    
    return wavpaths


def get_audio_id(audio_path):
    """Create unique id of audio sample by including leaf folder in the id."""
    temp = audio_path.split('/')[-2:] # [0] = subfolder, [1] = ____.wav
    temp[-1] = temp[-1].split('.wav')[0] if '.wav' in temp[-1] else temp[-1].split('.mp3')[0] if '.mp3' in temp[-1] else temp[-1].split('.flac')[0] # remove '.<extension>'
    return '/'.join(temp)


# common output ASR interface creation classes.

@dataclass()
class WordAlign:
    """Time-aligned word"""
    word: str  # the word itself.
    start_time: float  # start time of the word in seconds in the audio file.
    end_time: float  # end time of the word in seconds in the audio file.


@dataclass()
class Hypothesis:
    """A single hypothesis for an audio file."""
    pred_txt: str  # the natural language prediction as a string.
    word_aligns: Optional[List[WordAlign]] = field(default_factory=list) # optional list of WordAlign objects, which represent words with time alignments.


@dataclass()
class Hypotheses:
    """Hypotheses for one audio file."""
    hypotheses: List[Hypothesis]  # list of hypotheses produced by the ASR decoder. len=1 if only best hypothesis returned.


def write_results(results, wav_paths, out_dir):
    """
    Save the results of running ASR inference to multiple output hypotheses json files if more than one hypothesis is outputted per audio file, or save
     just the best hypotheses to a single output json file if only one hypothesis is outputted per audio file.
    The results are preprocessed into a common format that is framework-agnostic, as different ASR models are implemented in different frameworks,
     whose original outputs may be structured differently.
    Args:
        results (List[Hypotheses]):
            the common format for results for any ASR inference using the classes: Hypotheses, Hypothesis, WordAlign.
        wav_paths (List[str]):
            paths of audio files corresponding to Hypotheses objects in results.
        out_dir (str):
            the root output folder to which to save the output files.
    """

    def basic_item(wavpath, hypo):
        """Construct a basic 'item' dict with all fields present except for the optional 'timestamps_word' field."""
        item = dict()
        id = get_audio_id(wavpath)
        item['wav_path'] = wavpath
        item['id'] = id
        item['pred_txt'] = hypo.pred_txt

        return item

    for hypotheses, wav_path in zip(results, wav_paths):
        if len(hypotheses.hypotheses) > 1:
            # save ranks of hypotheses across the audio files to different output files.
            for i in range(len(hypotheses.hypotheses)):
                hypothesis = hypotheses.hypotheses[i]
                with open(os.path.join(out_dir, f"hypotheses{i+1}_of_{len(hypotheses.hypotheses)}.json"), 'a') as hyp_file:
                    # fill in main parts of the item dict.
                    item = basic_item(wav_path, hypothesis)

                    # if args.time_aligns
                    if hypotheses.hypotheses[i].word_aligns:
                        item['timestamps_word'] = [asdict(word_align) for word_align in hypothesis.word_aligns]

                    hyp_file.write(json.dumps(item) + "\n")
        else:
            # only one hypothesis per audio file, therefore save the best hypotheses across the audio files in a single output file.
            hypothesis = hypotheses.hypotheses[0]
            with open(os.path.join(out_dir, "best_hypotheses.json"), 'a') as hyp_file:
                # fill in main parts of the item dict.
                item = basic_item(wav_path, hypothesis)

                # if args.time_aligns
                if hypothesis.word_aligns:
                    item['timestamps_word'] = [asdict(word_align) for word_align in hypothesis.word_aligns]

                hyp_file.write(json.dumps(item) + "\n")
