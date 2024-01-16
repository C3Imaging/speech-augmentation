import os
import json
import copy
import shlex
import random
import subprocess
from itertools import product
from contextlib import ExitStack
from difflib import SequenceMatcher
import Utils.yaml_config as yaml_config
from dataclasses import dataclass, field
from typing import List, Union, Any, Optional, Tuple, Dict


@dataclass()
class Token:
    """ "A word-level natural language token with time alignment information."""

    token: str  # word-level string of the token itself.
    start_time: float  # start time of the token in seconds.
    end_time: float  # end time of the token in seconds.
    matched_flag: bool = False  # used for multi-ASR data funneling algorithm 'hypothesis_based_lengthwise_algo'.


@dataclass()
class Hypothesis:
    """A class representation of a single hypothesis consisting of multiple tokens with time alignment information."""

    tokens: List[
        Token
    ]  # the hypothesis itself, represented as a list of Token objects.
    weight: float  # depends on the Hypothesis's level. No need to scale by the total number of hypotheses considered, since regardless of how many are considered each weight is dependant only on its level.

    def get_match_score(self, token: Token) -> float:
        """Returns a similarity score between the first token from self.tokens to match in time with the input token, otherwise a score of 0.0 is returned indicating no tokens in self.tokens matched in time with the input token."""
        score = 0.0
        matched_token = self._find_match(token)
        if matched_token:
            matched_token.matched_flag = (
                True  # do not check this token again for future token inputs.
            )
            score = get_tokens_similarity_fuzzy(token.token, matched_token.token)
        return score

    def _find_match(self, token: Token) -> Union[Token, None]:
        """Returns Token from self.tokens that matches by time using an IOU range with the time from another Token from a different ASR decoder Hypothesis, otherwise if no match from self.tokens returns None."""
        matched_tok = None
        for tok in self.tokens:
            if not tok.matched_flag:
                time_match = IOU(
                    token,
                    tok,
                    threshold=yaml_config.cfg.get("iou_tokens_time_threshold"),
                )
                if time_match:
                    matched_tok = tok
                    break
        return matched_tok


@dataclass()
class ScoredToken(Token):
    # 'matched_flag' is the only parent property not of interest for this subclass.
    accum_score: float = 0  # accumulated score.


@dataclass()
class SpeechSegment:
    """A class representation of a contiguous segment of speech from an audio file."""

    seq_toks: List[ScoredToken] = field(
        default_factory=list
    )  # the speech segment itself represented as a sequence of tokens with scores.
    save_time_length: float = 2.0  # save segment if the total length of the sequence is this amount of time in seconds or more.

    def add(self, token: Token, score: float) -> None:
        """ "Add a token to the sequence of tokens if it doesn't yet exist in the sequence, otherwise add the score value to the accumulated score of that token."""
        # check if the token already exists in the sequence of tokens by IOU time.
        matched_tok = None
        for tok in self.seq_toks:
            time_match = IOU(
                token, tok, threshold=0.9
            )  # should be exact time match, but due to the use of floats its best to use IOU with a high threshold value
            if time_match:
                matched_tok = tok
                break
        if matched_tok:
            # if that token already exists, add to its accumulating score.
            matched_tok.accum_score += score
        else:
            # else append the token to the sequence list as the first time this token got a score.
            self.seq_toks.append(
                ScoredToken(
                    token=token.token,
                    start_time=token.start_time,
                    end_time=token.end_time,
                    accum_score=score,
                )
            )

    def try_save(self, audio_filepath: str, num_decoders: int) -> None:
        """If the segment is longer than a specified time length, funnel the corresponding <audio, text> parallel data into an output dataset folder based on the average confidence."""
        if self.seq_toks:
            total_time = (self.seq_toks[-1].end_time - self.seq_toks[0].start_time)  # total length of time of the segment in seconds.
            if total_time >= self.save_time_length:
                # funnel/save the resultant <audio, txt> pair for this speech segment into an output dataset, if the confidence is within the thresholds for some data funnel.
                try_save_speech_segment(audio_filepath, self)
                self.reset()

    def reset(self) -> None:
        """Clear the sequence of tokens."""
        self.seq_toks = list()

    def get_confidence(self, num_decoders: int) -> float:
        """Calculate overall average confidence of the sequence of scored tokens in the segment."""
        return sum(tok.accum_score for tok in self.seq_toks) / num_decoders


@dataclass()
class DecoderOutput:
    """A class representation of the hypotheses returned by a decoder for a single speech audio file."""

    # audio_filepath: str # the path to the audio file for which the hypotheses were created.
    hypotheses: List[Hypothesis]


def IOU(token1: Token, token2: Token, threshold: float) -> bool:
    times = [token1.start_time, token1.end_time, token2.start_time, token2.end_time]
    u = max(times) - min(
        times
    )  # the union in number of seconds between the two tokens.
    i = min([token1.end_time, token2.end_time]) - max(
        [token1.start_time, token1.start_time]
    )  # the intersection in number of seconds between the two tokens.
    # the two tokens match in time if the intersection/union is greater than the threshold ratio.
    time_match = True if i > threshold * u else False

    return time_match


def get_tokens_similarity_fuzzy(gt_token: str, hyp_token: str) -> float:
    """Fuzzy char similarity or lexicon-based similarity between two tokens, with token1 being the ground truth token, i.e. correct token in terms of lexical meaning."""
    return SequenceMatcher(None, gt_token, hyp_token).ratio()


def get_hypotheses_filenames(decoders_folderpaths: List[str]) -> List[str]:
    """Returns a list of hypotheses filepaths (to JSON files) which have the same names across all the decoder folders (a checked condition)."""
    hypotheses_files_decoders = (
        list()
    )  # each element is a list of hypotheses files extracted from a particular decoder folder.
    for folder_path in decoders_folderpaths:
        # get json hypotheses files.
        files = [
            file
            for file in os.listdir(folder_path)
            if "hypotheses" in file and file.endswith(".json")
        ]
        files.sort()
        if len(files) > 1:
            files = "/".join(files)
        hypotheses_files_decoders.append(files)
    # ensure filenames match across the folders.
    assert (
        len(set(hypotheses_files_decoders)) <= 1
    ), "ERROR: the number of hypotheses JSON files and their filenames must match across all decoder folders!!!"

    return hypotheses_files_decoders[0].split(
        "/"
    )  # assume at this stage that the filenames match across all decoder folders.


def check_wavpaths_match(
    decoders_folderpaths: List[str], hypotheses_filenames: List[str]
) -> None:
    """Ensures the set and order of wavpaths across the first hypotheses files of the decoder folders match (assume each hypotheses file in a particular decoder folder has the same set and order of wavpaths)."""
    wavpaths_decoders = list()
    # loop through the decoder folders.
    for root_path, hypotheses_file in zip(decoders_folderpaths, hypotheses_filenames):
        # assume all hypotheses.json files for a single decoder folder have the same set and order of wavpaths, therefore load wavpaths only from first file.
        with open(os.path.join(root_path, hypotheses_file), "r") as f:
            wavpaths = list()
            # loop through each line in the JSON file.
            for line in f:
                # load a line from the JSON file as a dict.
                item = json.loads(line)
                wavpaths.append(item["wav_path"])
            if len(wavpaths) > 1:
                # assume '?' is not a part of the filepath.
                wavpaths = "?".join(wavpaths)
            wavpaths_decoders.append(wavpaths)
    assert (
        len(set(wavpaths_decoders)) <= 1
    ), "ERROR: the set of 'wav_path' fields must be the same across the different decoder folders' hypotheses JSON files!!!"


def get_global_decoders_dict(
    decoders_folderpaths: List[str],
    hypotheses_filenames: List[str],
    hypothesis_level_weights: List[float],
) -> Dict[str, List[DecoderOutput]]:
    """Returns dictionary of hypotheses from different decoders across the global set of common audio filepaths for which the decoders have hypotheses.
    Format of 'global_decoders_out' dict returned:
      {
        '/path/to/audio1.wav': [DecoderOutput_decoder1, DecoderOutput_decoder2, ... , DecoderOutput_decoderN]
        '/path/to/audio2.wav': [DecoderOutput_decoder1, DecoderOutput_decoder2, ... , DecoderOutput_decoderN]
        ...
        '/path/to/audioM.wav': [DecoderOutput_decoder1, DecoderOutput_decoder2, ... , DecoderOutput_decoderN]
      }
    """
    global_decoders_out = dict()
    # loop through decoder folders.
    for root_path in decoders_folderpaths:
        with ExitStack() as stack:
            # read the same line from each hypotheses JSON file for this decoder.
            filenames = [
                os.path.join(root_path, hypotheses_file)
                for hypotheses_file in hypotheses_filenames
            ]
            files = [stack.enter_context(open(fname, "r")) for fname in filenames]
            for rows in zip(*files):
                # 'rows' is a tuple containing the same row from each hypotheses file.
                # in this case, it contains the decoder's hypotheses for the audio file on that row.

                # loop through the hypotheses for this decoder for this audio file.
                decoder_hyps = list()
                for hypothesis, weight in zip(rows, hypothesis_level_weights):
                    hyp_dict = json.loads(hypothesis)
                    # 1. create Token objects list for the current hypothesis.
                    tokens_list = [
                        Token(
                            token=word_dict["word"],
                            start_time=word_dict["start_time"],
                            end_time=word_dict["end_time"],
                        )
                        for word_dict in hyp_dict["timestamps_word"]
                    ]
                    # 2. create Hypothesis object and append to list of hypotheses for this audio file.
                    decoder_hyps.append(Hypothesis(tokens=tokens_list, weight=weight))
                # 3. create DecoderOutput object for this decoder for this audio file.
                decoder_out = DecoderOutput(hypotheses=decoder_hyps)
                # 4. add DecoderOutput object to the appropriate global dictionary based on the audio_filepath key.
                if global_decoders_out.get(hyp_dict["wav_path"]) is not None:
                    # if hypotheses were already added for this audio filepath by other decoders.
                    global_decoders_out.get(hyp_dict["wav_path"]).append(decoder_out)
                else:
                    # if this is the first time adding hypotheses for this audio filepath because we are on the first decoder folder.
                    global_decoders_out[hyp_dict["wav_path"]] = [decoder_out]

    return global_decoders_out


def get_longest_hypothesis_idx(hypotheses: List[Hypothesis]) -> int:
    """Returns the index of the longest (most amount of tokens) hypothesis from a list of Hypothesis objects. If more than one hypothesis has the max length, the index is select at random among these."""
    hyps_lens = [len(hypothesis.tokens) for hypothesis in hypotheses]
    max_len = max(
        hyps_lens
    )  # the maximum number of tokens (length) of a hypothesis from the input.
    max_idxs = [
        i for i, v in enumerate(hyps_lens) if v == max_len
    ]  # indexes of maximum length hypotheses.

    # randomly select the decoder index when there are multiple decoders whose level 1 hypothesis has max number of tokens.
    return random.choice(max_idxs)


def reset_all_tokens_flag(
    wavpath: str, decoders_dict: Dict[str, List[DecoderOutput]]
) -> None:
    """Resets the 'matched_flag' flag for each Token object for each Hypothesis object for each DecoderOuput object for the list of DecoderOutput objects for the particular audio file."""
    decoders = decoders_dict[wavpath]
    for decoder in decoders:
        for hypothesis in decoder.hypotheses:
            for token in hypothesis.tokens:
                token.matched_flag = False


def token_based_depthwise_algo(decoders_dict: Dict[str, List[DecoderOutput]]) -> None:
    """
    Args:
        decoders_dict:
          A dictionary with the following format:
            {
              '/path/to/audio1.wav': [DecoderOutput_decoder1, DecoderOutput_decoder2, ... , DecoderOutput_decoderN]
              '/path/to/audio2.wav': [DecoderOutput_decoder1, DecoderOutput_decoder2, ... , DecoderOutput_decoderN]
              ...
              '/path/to/audioM.wav': [DecoderOutput_decoder1, DecoderOutput_decoder2, ... , DecoderOutput_decoderN]
            }
    """
    # loop over the audio files and extract speech segments to save, if any.
    for wavpath, decoders_output_list in decoders_dict.items():
        ## select the longest, i.e. most amount of tokens, level 1 (best) hypothesis among the decoders as the 'driver' hypothesis against which others will be compared.
        # NOTE: the segments of speech will use the tokens and corresponding times from the 'driver' hypothesis.
        # get index of the decoder with the longest level 1 hypothesis (randomly selected if >1 decoder with the same max length), i.e. the 'driver'.
        # input is a list of Hypotheses, where each object is the best hypothesis of a decoder.
        driver_decoder_idx = get_longest_hypothesis_idx([decoder.hypotheses[0] for decoder in decoders_output_list])
        # extract the 'driver' hypothesis
        driver_hypothesis = decoders_output_list[driver_decoder_idx].hypotheses[0]
        # remove the 'driver' decoder from the list of decoders, resulting in a list of 'other' decoders, excluding the 'driver' decoder.
        # in-place removal saves memory instead of deep copying and then removing.
        del decoders_output_list[driver_decoder_idx]

        # initialise an empty current speech segment object for this audio file.
        current_speech_segment = SpeechSegment(save_time_length=yaml_config.cfg.get("speech_segment_length"))
        # loop over the tokens of the driver hypothesis.
        for token in driver_hypothesis.tokens:
            match = False
            # loop over the 'other' decoders.
            for other_decoder in decoders_output_list:
                # loop over each hypothesis of the current 'other' decoder.
                for hypothesis in other_decoder.hypotheses:
                    # get the match score for the current 'driver' token with the current hypothesis of the current 'other' decoder.
                    score = hypothesis.get_match_score(token)
                    # multiply the simple similarity score by the current hypothesis' weight (weight depends on level).
                    score *= hypothesis.weight
                    if score >= yaml_config.cfg.get("compound_score_threshold"):
                        match = True  # the current 'driver' token matched with a token from at least one 'other' decoder's hypothesis at any level.
                        # add the current 'driver' token to the speech segment.
                        current_speech_segment.add(token, score)
                        break  # no need to check for matches in lower hypothesis levels.
            if not match:
                # no matches at all for the current 'driver' token across all 'other' decoders, i.e. a gap in the continuity of the current speech segment.
                current_speech_segment.reset()
            current_speech_segment.try_save(wavpath, len(decoders_output_list))


def hypothesis_based_lengthwise_algo(
    decoders_dict: Dict[str, List[DecoderOutput]]
) -> None:
    """
    Args:
        decoders_dict:
          A dictionary with the following format:
            {
              '/path/to/audio_1.wav': [DecoderOutput_decoder1, DecoderOutput_decoder2, ... , DecoderOutput_decoderN]
              '/path/to/audio_2.wav': [DecoderOutput_decoder1, DecoderOutput_decoder2, ... , DecoderOutput_decoderN]
              ...
              '/path/to/audio_M.wav': [DecoderOutput_decoder1, DecoderOutput_decoder2, ... , DecoderOutput_decoderN]
            }
    """

    def process_hypotheses_combination(
        hypotheses: List[Hypothesis],
    ) -> List[SpeechSegment]:
        """Returns a list of SpeechSegment objects representing the extracted speech segments found in the combination of inputted hypotheses."""
        speech_segments = (
            list()
        )  # a list of speech segments extacted from the current combination of hypotheses.
        ## select the longest, i.e. most amount of tokens, hypothesis as the 'driver' hypothesis against which others will be compared.
        # NOTE: the segments of speech will use the tokens and corresponding times from the 'driver' hypothesis.
        driver_hyp_idx = get_longest_hypothesis_idx(hypotheses)
        driver_hypothesis = hypotheses[driver_hyp_idx]
        # remove the 'driver' hypothesis from the current combination of hypotheses, resulting in a list of 'other' hypotheses for the 'other' decoders, excluding the 'driver' decoder.
        # in-place removal saves memory instead of deep copying and then removing.
        del hypotheses[driver_hyp_idx]
        # initialise an empty current speech segment object for this audio file.
        current_speech_segment = SpeechSegment(
            save_time_length=yaml_config.cfg.get("speech_segment_length")
        )
        # loop over the tokens of the 'driver' hypothesis.
        for token in driver_hypothesis.tokens:
            match = False
            # loop over each 'other' hypothesis.
            for other_hyp in hypotheses:
                # get the match score for the current 'driver' token with a (ONLY ONE) token from the current 'other' hypothesis IF the times match (otherwise score will be 0.0).
                score = other_hyp.get_match_score(token)
                # multiply the simple similarity score by the 'driver' hypothesis' weight and the current 'other' hypothesis' weight (weight depends on level).
                score *= driver_hypothesis.weight * other_hyp.weight
                if score >= yaml_config.cfg.get("compound_score_threshold"):
                    match = True  # the current 'driver' token matched with a token from at least one 'other' hypothesis.
                    current_speech_segment.add(token, score)
            if not match:
                # no matches at all for the current 'driver' token across all 'other' hypotheses, i.e. a gap in the continuity of the current speech segment.
                current_speech_segment.reset()
            if current_speech_segment.seq_toks:
                total_time = (
                    current_speech_segment.seq_toks[-1].end_time
                    - current_speech_segment.seq_toks[0].start_time
                )  # total length of time of the segment in seconds.
                if total_time >= current_speech_segment.save_time_length:
                    speech_segments.append(copy.deepcopy(current_speech_segment))
                    current_speech_segment.reset()

        return speech_segments

    # initialise a dictionary with the following format:
    #         {
    #           '/path/to/audio_1.wav': [[SpeechSegment_1_1, SpeechSegment_1_2, ..., SpeechSegment_1_M], [SpeechSegment_2_1, SpeechSegment_2_2, ..., SpeechSegment_2_M], ... , [SpeechSegment_K_1, SpeechSegment_K_2, ..., SpeechSegment_K_M]]
    #           '/path/to/audio_2.wav': [[SpeechSegment_1_1, SpeechSegment_1_2, ..., SpeechSegment_1_M], [SpeechSegment_2_1, SpeechSegment_2_2, ..., SpeechSegment_2_M], ... , [SpeechSegment_K_1, SpeechSegment_K_2, ..., SpeechSegment_K_M]]
    #           ...
    #           '/path/to/audio_N.wav': [[SpeechSegment_1_1, SpeechSegment_1_2, ..., SpeechSegment_1_M], [SpeechSegment_2_1, SpeechSegment_2_2, ..., SpeechSegment_2_M], ... , [SpeechSegment_K_1, SpeechSegment_K_2, ..., SpeechSegment_K_M]]
    #         }
    #  Each value of the dictionary is a list of lists of SpeechSegment objects.
    #  The innermost list represents all the speech segments extracted from that audio file based on a particular combination of hypotheses from different decoders (one from each decoder).
    #  The list of lists represents all speech segments extracted across all possible combinations of hypotheses across the decoders.
    all_audiofiles_all_speech_segments = dict()

    # loop over the audio files and extract speech segments to save, if any.
    for wavpath, decoders_output_list in decoders_dict.items():
        # initialise a list of lists. Represents ALL possible speech segments for this audio file when considering ALL possible combinations of N hypotheses (one hypothesis from each decoder),
        #  where N=number of decoders.
        all_speech_segments = list()
        # initialise an empty current speech segment object for this audio file.
        current_speech_segment = SpeechSegment(
            save_time_length=yaml_config.cfg.get("speech_segment_length")
        )
        all_hypotheses = [decoder.hypotheses for decoder in decoders_output_list]
        # loop over each possible combination of N hypotheses and each element is a hypothesis from a different decoder.
        for hypotheses_combination in product(*all_hypotheses):
            speech_segments = process_hypotheses_combination(
                list(hypotheses_combination)
            )
            # if at least one speech segments was extracted from the current combination of hypotheses for the current audio file.
            if speech_segments:
                all_speech_segments.append(speech_segments)
            # reset all Token's matched_flag properties for all Hypothesis objects for all DecoderOutput objects for the current wavpath.
            reset_all_tokens_flag(wavpath, decoders_dict)
        # if at least one combination of hypotheses extracts at least one speech segment, else add an empty list for that audio file (no speech segments extracted at all across all combinations of hypotheses).
        all_audiofiles_all_speech_segments[wavpath] = (
            all_speech_segments if all_speech_segments else []
        )

    # for each audio file, select the list of speech segments (representing a combination of hypotheses) that has the highest average confidence value across the speech segments and save them.
    for wavpath, speech_segments_list in all_audiofiles_all_speech_segments.items():
        # a list of average confidence values across the list of speech segment lists for this audio file.
        # i.e. a value per each list of speech segments (i.e. per each combination of hypotheses).
        confidences = [
            sum(
                [
                    speech_segment.get_confidence(
                        num_decoders=len(speech_segments_list) - 1 # divide by the number of 'other' decoders (i.e. excluding the 'driver').
                    )
                    for speech_segment in speech_segments
                ]
            )
            for speech_segments in speech_segments_list
        ]
        max_conf = max(confidences)  # the maximum confidence.
        max_idxs = [i for i, v in enumerate(confidences) if v == max_conf]  # indexes of maximum confidence speech segments lists.
        # randomly select the list of speech segments when there are multiple speech segments lists whose average confidence is the maximum.
        speech_segments_list_idx = random.choice(max_idxs)
        # save the selected list of speech segments for this audio file.
        # decide where to save the resultant <audio, txt> pair based on the confidence score.
        for speech_segment in speech_segments_list[speech_segments_list_idx]:
            try_save_speech_segment(wavpath, speech_segment)
                    
                    
def try_save_speech_segment(in_wavpath: str, speech_segment: SpeechSegment) -> None:
    """Save each speech segment as an <audio, txt> pair, extracted from the audio file,
        to an output folder based on the confidence of the segment, if the confidence is within the thresholds for data funnels.
    """
    def save_speech_segment():
        global output_segments_filenames
        in_filename = in_wavpath.split('/')[-1].split(".wav")[0]
        if not output_segments_filenames.get(in_filename):
            # if no speeech segments were saved from this audio file yet, initialise an entry for this audio file.
            output_segments_filenames[in_filename] = 0
        # get the current amount of speech segments saved from this audio file.
        cur_saved_count = output_segments_filenames[in_filename]
        # create a unique output filename for this speech segment.
        output_filename = in_filename + f"_{cur_saved_count+1}"
        # get the common full path for audio and corresponding text without the file extension.
        output_filepath = os.path.join(threshold_entry[1], output_filename)
        # save the speech segment to a wav file by extracting it from the corresponding audio file.
        start_time = speech_segment.seq_toks[0].start_time
        subprocess.run(shlex.split(f"ffmpeg -y -ss {start_time} -i {in_wavpath} -t {speech_segment.seq_toks[-1].end_time - start_time} {output_filepath}.wav"))
        # save the corresponding speech segment's text data to a separate txt file with the same filename.
        with open(f"{output_filepath}.txt", "w") as f:
            f.write(" ".join([tok.token for tok in speech_segment.seq_toks]))
        # increment the number of saved speech segments from this audio file.
        output_segments_filenames[in_filename] += 1
    
    avg_conf = speech_segment.get_confidence(num_decoders=len(yaml_config.cfg.get('decoder_folders'))-1) # num_decoders = number of 'other' decoders, i.e. excluding the 'driver'.
    # funnel the speech segments into the appropriate output parallel audio dataset directory according to its average confidence value.
    for threshold_entry in yaml_config.cfg.get("output_subfolders"):
        # 'threshold_entry' is a tuple where:
        # [0] = either a tuple of 2 threshold values or a single threshold value.
        # [1] = output folder for this threshold(s).
        if type(threshold_entry[0]) == tuple:
            # if the average confidence value of the tokens in a speech segment is between two thresholds.
            if avg_conf >= threshold_entry[0][0] and avg_conf < threshold_entry[0][1]:
                save_speech_segment()
        else:
            # if the average confidence value of the tokens in a speech segment is more than the max threshold.
            if avg_conf >= threshold_entry[0]:
                save_speech_segment()


output_segments_filenames = dict()
