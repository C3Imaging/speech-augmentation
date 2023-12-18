"""
Uses principles of dependency injection and dependency inversion and creating factories to simplify API of using wav2vec2 models loaded in different ways + various decoders for ASR inference.
"""


import re
import os
import math
import torch
import logging
import fileinput
import torchaudio
from tqdm import tqdm
from argparse import Namespace
from omegaconf import OmegaConf
from abc import ABC, abstractmethod
from . import decoding_utils_chkpt
from fairseq.data import Dictionary
from typing import List, Any, Union, Optional, Tuple, Dict

if __name__ == "__main__":
    import Utils.asr.decoding_utils_chkpt as decoding_utils_chkpt, Utils.asr.decoding_utils_torch as decoding_utils_torch
else:
    from . import decoding_utils_torch

from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc, Wav2Vec2CtcConfig
from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder, W2lKenLMDecoder, W2lFairseqLMDecoder


def convert_to_upper(filepath):
    """Utility script that converts words in a vocab file to upper case and overwrites the file line by line."""
    # skip conversion if words are already upper case (assume that if the first line's word is upper cased, then all lines are)
    f = open(filepath, 'r')
    first_word = f.readline().split(' ')[0]
    if first_word.isupper():
        f.close()
        return False
    f.close()
            
    # Opening a file with fileinput.input while specifying inplace=True copies the original input file to a backup file before being erased of its contents,
    #  so a check is performed prior to opening wth fileinput.
    with fileinput.input(files=(filepath), inplace=True) as file:
        for line in file:
            # assumes format of vocab file has word as the first element on a line and elements are separated by a space
            word_and_freq = line.split(' ')
            
            if word_and_freq[0].isupper():
                break
            
            word_and_freq[0] = word_and_freq[0].upper()
            new_line = ' '.join(word_and_freq)
            # By specifying inplace=True, the standard output is redirected to the original file which is now considered the output file,
            #  thus enabling in place overwriting.
            # The backup file is deleted when the output file is closed.
            print('{}'.format(new_line), end='') # end='' suppresses the addition of a newline character as the line already contains one
    return True


def get_word_time_alignments_torch(audio_len, num_frames, sample_rate, tokens, timesteps):
    """
    Args:
        audio_len (int):
            The length of audio file in number of samples.
        num_frames (int):
            The number of frames in the ASR acoustic model emission matrix.
        sample_rate (int):
            The sample rate of the loaded audio file.
        tokens (List[str]):
            Decoded list of characters corresponding to the non-blank tokens returned by the decoder.
        timesteps (torch.tensor[int]):
            Frame numbers corresponding to the non-blank tokens.

    Returns:
        word_times (List[Tuple[float, float]]):
            List of tuples of start_time and stop_time in seconds for word in the transcript.
    """
    ratio = audio_len / num_frames / sample_rate
    chars = []
    words = []
    word_start = None
    for token, timestep in zip(tokens, timesteps * ratio):
        if token == "|":
            if word_start is not None:
                words.append((word_start, timestep))
            word_start = None
        else:
            chars.append((token, timestep))
            if word_start is None:
                word_start = timestep

    word_times = [(w_start.item(), w_end.item()) for w_start, w_end in words]

    return word_times


def get_word_time_alignments_fairseq(audio_len, num_frames, sample_rate, symbols, timesteps):
    """Get word time alignments information for a hypothesis transcript input by converting from timesteps to seconds.
    Args:
        audio_len (int):
            The length of audio file in number of samples.
        num_frames (int):
            The number of frames in the ASR acoustic model emission matrix.
        sample_rate (int):
            The sample rate of the loaded audio file.
        symbols (List[str]):
            Decoded list of characters corresponding to the non-blank tokens returned by the decoder.
        timesteps (List[int]):
            Frame numbers corresponding to the non-blank tokens/symbols.

    Returns:
        word_times (List[Tuple[float, float]]):
            List of tuples of start_time and stop_time in seconds for word in the transcript.
    """
    # list of times in seconds in the corresponding audio file for the the non-blank tokens/symbols.
    timestamps = []
    # get the timestep in seconds corresponding to each non-blank token.
    for frame_num in timesteps:
        timestamp = frame_num * (audio_len / (num_frames * sample_rate))
        timestamps.append(timestamp)

    # NOTE: algorithm only works if the first and last symbols are '|', so add them in if that's not the case.
    frame_offset = 0
    if symbols[0] != '|':
        symbols.insert(0, '|')
        # if adding a symbol at index 0, all symbols will have their frame idx increased by 1, so an offset of -1 is created.
        frame_offset = -1
    if symbols[-1] != '|':
        symbols.append('|')

    word_boundary_idxs = [] # tuples of word start and stop indices.
    # get the indices of all word-boundary tokens (|).
    wb_tokens_idxs = [i for i in range(len(symbols)) if symbols[i] == '|']

    # create tuples for each word that contains the indices of its start symbol and end symbol.
    tup = [] # initialise the first tuple of word start character and word end character indices.
    # loop through the indices of the '|' tokens and find the indices of the word-boundary symbols/characters that are the start and end characters of each word.
    for wb_tokens_idx in wb_tokens_idxs:
        try:
            if symbols[wb_tokens_idx-1] != '|' and tup:
                # there is a start index in tuple, but no end index yet.
                # end index has been found.
                if wb_tokens_idx-1 == tup[0]:
                    # word is composed of only one character, add the index of this '|' token as the end character index for the word.
                    tup.append(wb_tokens_idx)
                else:
                    # word is composed of more than one character.
                    tup.append(wb_tokens_idx-1) # add an end character index for the word.
                # add the tuple as complete word to the list of word start and end index tuples.
                word_boundary_idxs.append(tup)
                tup = [] # reset the tuple.
                # continue onto the next if statement as this '|' token may be the boundary between two words.
            if symbols[wb_tokens_idx+1] != '|':
                # start character of new word reached.
                tup.append(wb_tokens_idx+1) # add a start character index for the word.
        except IndexError:
            continue
    
    # create tuples of start and stop times for each word
    word_times = [(timestamps[start_idx + frame_offset], timestamps[end_idx + frame_offset]) for start_idx, end_idx in word_boundary_idxs]

    return word_times


def normalize_timestamp_output_w2v2(words, word_time_tuples):
    """Get word Dict objects with time information for each word in the hypothesis transcript.

    Args:
        words (List[str]):
            List of words in the transcript.
        word_time_tuples (List[Tuple[float,float]]):
            List of tuples of start_time and stop_time in seconds for word in the transcript.

    Returns:
        values (List[Dict]):
            List of dict objects where each dict has the following fields:
                'word': (str) the word itself.
                'start_time': (float) the start time in seconds of the word in the corresponding audio file.
                'end_time': (float) the end time in seconds of the word in the corresponding audio file.
    """
    values = []
    for word, (word_start, word_end) in zip(words, word_time_tuples):
        vals_dict = dict()
        vals_dict['word'] = word
        vals_dict['start_time'] = word_start
        vals_dict['end_time'] = word_end
        values.append(vals_dict)
    
    return values


def beam_search_decode_torch(decoder, hypos, emission_mx, audio_lens, num_hyps, time_aligns):
    transcripts = []

    for i in range(emission_mx.size(dim=0)):
        # if the batch_size is > 1, use the maximum original audio length in the batch, as all other audio files are padded to the max length during preprocessing.
        audio_len = audio_lens[i] if emission_mx.size(dim=0) == 1 else max(audio_lens)
        # append a list of all hypotheses for this element of the batch.
        if num_hyps > 1:
            all_results = []
            for hyp in hypos[i]:
                hyp_dict = dict()
                if hyp.words:
                    # 'words' instance variable is not empty if using a lexicon.
                    transcript = ' '.join(hyp.words).lower()
                else:
                    # 'words' instance variable is [] if lexicon-free decoding, convert from non-blank tokens to words instead.
                    tokens = hyp.tokens
                    tokens_str = ''.join(decoder.idxs_to_tokens(tokens))
                    transcript = ' '.join(tokens_str.split('|')).strip().lower()
                hyp_dict['pred_txt'] = transcript
                if time_aligns:
                    word_times = get_word_time_alignments_torch(audio_len, emission_mx.size(dim=1), 16000, decoder.idxs_to_tokens(hyp.tokens), hyp.timesteps)
                    timestamps_word = normalize_timestamp_output_w2v2(transcript.split(' '), word_times)
                    hyp_dict['timestamps_word'] = timestamps_word
                    
                # add a hypothesis dict
                all_results.append(hyp_dict)

            transcripts.append(all_results)
        else:
            hyp_dict = dict()
            hyp = hypos[i][0]
            # append the decoded phrase (as a list of words) from the prediction of the first beam [0] (most likely transcript).
            if hyp.words:
                # 'words' instance variable is not empty if using a lexicon.
                transcript = ' '.join(hyp.words).lower()
            else:
                # 'words' instance variable is [] if lexicon-free decoding, convert from non-blank tokens to words instead.
                tokens = hyp.tokens
                tokens_str = ''.join(decoder.idxs_to_tokens(tokens))
                transcript = ' '.join(tokens_str.split('|')).strip().lower()
            hyp_dict['pred_txt'] = transcript
            if time_aligns:
                word_times = get_word_time_alignments_torch(audio_len, emission_mx.size(dim=1), 16000, decoder.idxs_to_tokens(hyp.tokens), hyp.timesteps)
                timestamps_word = normalize_timestamp_output_w2v2(transcript.split(' '), word_times)
                hyp_dict['timestamps_word'] = timestamps_word

            # add a hypothesis dict
            transcripts.append(hyp_dict)

    return transcripts


def beam_search_decode_fairseq(hypos, emission_mx, audio_lens, num_hyps, time_aligns):
    """Process the results of a W2lDecoder object from fairseq.

    Args:
        hypos (Union[List[Dict], List[List[Dict]]]):
            List of results for each audio file returned by a W2lDecoder object. If the number of hypotheses to return (W2lDecoder.nbest) is 1, hypos will be a list of just the best hypotheses dicts.
             If W2lDecoder.nbest > 1, hypos will be a list of lists, where for each audio file there will be N best hypotheses dicts.
        emission_mx (torch.tensor(B,T,N)):
            The batched emission matrix outputted by the w2v2 acoustic model trained in fairseq.
        audio_lens (List[int]):
            The lengths of the original audio files in the batch, measured in number of samples.
        num_hyps (int):
            The number of best hypotheses to return per audio file.
        time_aligns (bool):
            Flag used to specify whether to calculate word-level time alignment in seconds for each hypothesis.

    Returns:
        transcripts (Union[List[Dict], List[List[Dict]]]):
            List of processed results for each audio file. If W2lDecoder.nbest = 1, transcripts will be a list of just the best hypotheses dicts.
             If W2lDecoder.nbest > 1, transcripts will be a list of lists, where for each audio file there will be N best hypotheses dicts.
            A hypothesis dict has the following fields:
                'pred_txt': (str) the transcript hypothesis itself.
                'timestamps_word': (List[Dict]) List of word Dict objects, one for each word in the transcript, with the following fields:
                    'word': the word itself.
                    'start_time': the start time of the word in seconds in the corresponding audio file.
                    'end_time': the end time of the word in seconds in the corresponding audio file.
    """
    transcripts = []
    for i in range(emission_mx.size(dim=0)):
        # if the batch_size is > 1, use the maximum original audio length in the batch, as all other audio files are padded to the max length during preprocessing.
        audio_len = audio_lens[i] if emission_mx.size(dim=0) == 1 else max(audio_lens)
        if num_hyps > 1:
            all_results = []
            for hyp in hypos[i]:
                hyp_dict = dict()
                if hyp['words']:
                    # 'words' field is not empty if using a lexicon.
                    transcript = ' '.join(hyp['words']).lower()
                else:
                    # 'words' field is [] if lexicon-free decoding, convert from non-blank symbols to words instead.
                    tokens_str = ''.join(hyp['symbols'])
                    transcript = ' '.join(tokens_str.split('|')).strip().lower()
                hyp_dict['pred_txt'] = transcript
                if time_aligns:
                    word_times = get_word_time_alignments_fairseq(audio_len, emission_mx.size(dim=1), 16000, hyp['symbols'], hyp['timesteps'])
                    timestamps_word = normalize_timestamp_output_w2v2(hyp_dict['pred_txt'].split(' '), word_times)
                    hyp_dict['timestamps_word'] = timestamps_word
                # add a hypothesis dict
                all_results.append(hyp_dict)
                
            transcripts.append(all_results)
        else:
            hyp_dict = dict()
            # append the decoded phrase (as a list of words) from the prediction of the first beam [0] (most likely transcript).
            if hypos[i][0]['words']:
                # 'words' field is not empty if using a lexicon.
                transcript = ' '.join(hypos[i][0]['words']).lower()
            else:
                # 'words' field is [] if lexicon-free decoding, convert from non-blank symbols to words instead.
                tokens_str = ''.join(hypos[i][0]['symbols'])
                transcript = ' '.join(tokens_str.split('|')).strip().lower()
            hyp_dict['pred_txt'] = transcript
            if time_aligns:
                word_times = get_word_time_alignments_fairseq(audio_len, emission_mx.size(dim=1), 16000, hypos[i][0]['symbols'], hypos[i][0]['timesteps'])
                timestamps_word = normalize_timestamp_output_w2v2(hyp_dict['pred_txt'].split(' '), word_times)
                hyp_dict['timestamps_word'] = timestamps_word
            # add a hypothesis dict
            transcripts.append(hyp_dict)

    return transcripts


# wav2vec2 ASR model abstract class
class BaseWav2Vec2Model(ABC):
    # show what instance attributes should be defined
    model: torch.nn.Module
    vocab_path_or_bundle: str
    sample_rate: int

    def __init__(self, vocab_path_or_bundle: str) -> None:
        """Constructor for a model from a checkpoint (.pt) filepath or a torchaudio bundle."""
        self.vocab_path_or_bundle = vocab_path_or_bundle
    
    @abstractmethod
    def forward(self, filepaths: List[str], device: torch.device) -> Tuple[torch.Tensor, List[int]]:
        """Runs inference on a batch of audio samples, first preprocessing the audio as required by the model, and then returning the emissions matrices in a batch.
        All implementations return the same format of output emissions and audio lengths, regardless of the framework that the ASR model was loaded from.
        Returns:
            emissions (torch.tensor):
                A batched tensor of emission matrices generated by the ASR acoustic model.
            lengths (List[int]):
                A list of original audio lengths in number of samples after loading the audio files data. If batch_size=1 there will only be 1 element in the list.
        """

    def _preproc(self, filepaths: List[str], target_sr: int = 16000) -> List[torch.Tensor]:
        # common preprocessing steps for wav files
        waveforms, sr = decoding_utils_chkpt.get_features_list(filepaths)
        if sr != target_sr:
            waveforms = [torchaudio.functional.resample(waveform, sr, target_sr) for waveform in waveforms]

        return waveforms


# wav2vec2 ASR models (concrete implementations)
class TorchaudioWav2Vec2Model(BaseWav2Vec2Model):
    def __init__(self, device: torch.device, vocab_path_or_bundle: str = '') -> None:
        super().__init__(vocab_path_or_bundle)
        # if the passed in string is a torchaudio bundle
        assert re.match(r'torchaudio.pipelines', vocab_path_or_bundle) is not None, "ERROR!!! Please specify the torchaudio bundle to use as this wav2vec2 model."
        bundle = eval(vocab_path_or_bundle)
        self.model = bundle.get_model().to(device)
        self.sample_rate = bundle.sample_rate

    def forward(self, filepaths: List[str], device: torch.device) -> Tuple[torch.Tensor, List[int]]:
        """can perform batched inference."""
        waveforms = self._preproc(filepaths, self.sample_rate)

        padded_features, padding_masks = decoding_utils_chkpt.get_padded_batch_mxs(waveforms)
        lengths = torch.tensor([waveform.shape[0] for waveform in waveforms]) # original lengths in samples of the audio files.

        padded_features = padded_features.to(device)
        lengths = lengths.to(device)

        emissions, num_valid_frames = self.model(padded_features, lengths)

        return emissions, lengths.tolist()


class ArgsWav2Vec2Model(BaseWav2Vec2Model):
    def __init__(self, device: torch.device, model_filepath: str, vocab_path_or_bundle: str) -> None:
        super().__init__(vocab_path_or_bundle)

        assert re.match(r'torchaudio.pipelines', vocab_path_or_bundle) is None, "ERROR!!! Cannot specify a torchaudio bundle as this wav2vec2 model's vocab, please specify a path to a txt file instead."
        w2v = torch.load(model_filepath)

        assert w2v['args'] is not None, "ERROR!!! This checkpoint file does not have an args field defined in its loaded dict!"
        args_dict = decoding_utils_chkpt.get_config_dict(w2v['args'])
        w2v_config_obj = Wav2Vec2CtcConfig(**args_dict)

        target_dict = Dictionary.load(self.vocab_path_or_bundle)
        dummy_target_dict = {'target_dictionary' : target_dict.symbols}
        dummy_target_dict = Namespace(**dummy_target_dict)

        # in the fairseq framework, the vocab is needed for building the ASR model
        self.model = Wav2VecCtc.build_model(w2v_config_obj, dummy_target_dict)
        self.model.load_state_dict(w2v["model"], strict=True)
        self.model = self.model.to(device)
        self.model.eval()
        self.sample_rate = w2v['args'].sample_rate

    def forward(self, filepaths: List[str], device: torch.device) -> Tuple[torch.Tensor, List[int]]:
        """Inference is not called directly in the fairseq framework API but is done through a decoder, but I copy-pasted the actual inference part from
        fairseq/examples/speech_recognition/w2l_decoder.W2lDecoder.generate()"""
        waveforms = self._preproc(filepaths, self.sample_rate)
        lengths = torch.tensor([waveform.shape[0] for waveform in waveforms])
        padded_features, padding_masks = decoding_utils_chkpt.get_padded_batch_mxs(waveforms)
        padded_features = padded_features.to(device)

        sample, input = dict(), dict()
        input["source"] = padded_features
        input["padding_mask"] = padding_masks
        sample["net_input"] = input

        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }

        # Run encoder and normalize emissions
        encoder_out = self.model(**encoder_input)

        if hasattr(self.model, "get_logits"):
            emissions = self.model.get_logits(encoder_out) # no need to normalize emissions
        else:
            emissions = self.model.get_normalized_probs(encoder_out, log_probs=True)
        emissions = emissions.transpose(0, 1).float().cpu().contiguous()

        return emissions, lengths.tolist()


class CfgWav2Vec2Model(BaseWav2Vec2Model):
    def __init__(self, device: torch.device, model_filepath: str, vocab_path_or_bundle: str) -> None:
        super().__init__(vocab_path_or_bundle)

        assert re.match(r'torchaudio.pipelines', vocab_path_or_bundle) is None, "ERROR!!! Cannot specify a torchaudio bundle as this wav2vec2 model's vocab, please specify a path to a txt file instead."
        w2v = torch.load(model_filepath)

        assert w2v.get('cfg', None) is not None, "ERROR!!! This checkpoint file does not have a cfg field defined in its loaded dict!"
        args_dict = decoding_utils_chkpt.get_config_dict(w2v['cfg']['model'])
        w2v_config_obj = OmegaConf.merge(OmegaConf.structured(Wav2Vec2CtcConfig), args_dict)

        target_dict = Dictionary.load(self.vocab_path_or_bundle)
        dummy_target_dict = {'target_dictionary' : target_dict.symbols}
        dummy_target_dict = Namespace(**dummy_target_dict)

        # in the fairseq framework, the vocab is needed for building the ASR model
        self.model = Wav2VecCtc.build_model(w2v_config_obj, dummy_target_dict)
        self.model.load_state_dict(w2v["model"], strict=True)
        self.model = self.model.to(device)
        self.model.eval()
        self.sample_rate = w2v['cfg']['task']['sample_rate']

    def forward(self, filepaths: List[str], device: torch.device) -> Tuple[torch.Tensor, List[int]]:
        """Inference is not called directly in the fairseq framework API but is done through a decoder, but I copy-pasted the actual inference part from
        fairseq/examples/speech_recognition/w2l_decoder.W2lDecoder.generate()"""
        waveforms = self._preproc(filepaths, self.sample_rate)
        lengths = torch.tensor([waveform.shape[0] for waveform in waveforms])
        padded_features, padding_masks = decoding_utils_chkpt.get_padded_batch_mxs(waveforms)
        padded_features = padded_features.to(device)

        sample, input = dict(), dict()
        input["source"] = padded_features
        input["padding_mask"] = padding_masks
        sample["net_input"] = input

        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }

        # Run encoder and normalize emissions
        encoder_out = self.model(**encoder_input)

        if hasattr(self.model, "get_logits"):
            emissions = self.model.get_logits(encoder_out) # no need to normalize emissions
        else:
            emissions = self.model.get_normalized_probs(encoder_out, log_probs=True)
        emissions = emissions.transpose(0, 1).float().cpu().contiguous()

        return emissions, lengths.tolist()


# ASR decoder abstract class
class BaseDecoder(ABC):
    # show what instance attributes should be defined
    decoder: Any # different decoders do not have a common interface.
    num_hyps: int # if the decoder is beam search, specify how many of the best hypotheses to return, if 1 -> just the best hypothesis returned.
    time_aligns: bool # flag specifying whether to save word-level time alignments along with the transcript hypothesis/hypotheses.

    @abstractmethod
    def generate(self, emission_mx: torch.Tensor, audio_lens: Optional[List[int]] = None) -> Union[List[Dict], List[List[Dict]]]:
        """Generates a list of hypothesis dicts by decoding the output batch of a wav2vec2 ASR acoutic model.
        All implementations use the same format for hypothesis dict objects and return object, regardless of the decoder object's framework.

        Args:
            emission_mx (torch.tensor):
                A batched tensor of emission matrices generated by the ASR acoustic model.
            audio_lens (List[int]):
                A list of original audio lengths in the batch, measured in the number of samples after loading the audio files data.
        Returns:
            transcripts (Union[List[Dict], List[List[Dict]]]):
                If self.num_hyps=1 (i.e. return just the best hypothesis):
                    Then a list of hypothesis objects will be returned, one per audio file in batch (if batch_size=1, a list is still returned, just with 1 element).
                If self.num_hyps>1:
                    For decoders that support returning multiple hypotheses, a list of lists is returned, where there is a list for each audio file in batch, 
                    and for each audio file the list contains hypothesis dict objects corresponding to the top self.num_hyps hypotheses. 
                    (if batch_size=1, a list of lists is still returned, just with 1 element (which is a list of hypothesis dict objects)).
                If self.time_aligns:
                    For decoders that support time alignment, the hypothesis dict objects will have a 'timestamps_word' field.

        Hypothesis dict objects have the following fields:
            'pred_txt': [str] the sentence version of the transcript.
            'timestamps_word': [List[dict]] (Optional -> if self.time_aligns=True) a list of dict objects that contain start times and stop times in seconds
                in the corresponding audio file for each word in 'pred_txt'. The dict objects have the following fields:
                    'word': [str] the word itself.
                    'start_time' [float] the start time in seconds for that word.
                    'end_time' [float] the end time in seconds for that word.
        """

    @staticmethod
    def _get_vocab(vocab_path_or_bundle: str) -> List[str]:
        # if the passed in string is a torchaudio bundle
        if re.match(r'torchaudio.pipelines', vocab_path_or_bundle):
            bundle = eval(vocab_path_or_bundle)
            vocab =  [label.lower() for label in bundle.get_labels()]
            return vocab
        # else the passed in string is a text file with a vocabulary, load it using fairseq
        target_dict = Dictionary.load(vocab_path_or_bundle)
        return [c.lower() for c in target_dict.symbols]


# ASR decoders (concrete implementations)
class GreedyDecoder(BaseDecoder):
    """This algorithm does not explore all possibilities: it takes a sequence of local, hard decisions (about the best question to split the data)
    and does not backtrack. It does not guarantee to find the globally-optimal tree.

    Greedy decoder implementation here is loaded from torchaudio.
    """
    def __init__(self, vocab_path_or_bundle: str, time_aligns: Optional[bool]=None) -> None:
        # the vocabulary of chars known by the acoustic model that will be used for decoding
        # vocab is passed to the decoder object during initialisation
        vocab = self._get_vocab(vocab_path_or_bundle)
        self.decoder = decoding_utils_torch.GreedyCTCDecoder(vocab)
        # print decoder info.
        logging.info("------ Decoder ------")
        logging.info("Greedy Decoder from torchaudio.")
        logging.info(f"Type: {self.decoder}")
        logging.info(f"Vocab: {self.decoder.labels}")
        logging.info("Returns best hypotheses: 1")
        logging.info("---------------------")

    def generate(self, emission_mx: torch.Tensor, audio_lens: Optional[List[int]] = None) -> List[Dict]:
        # generate one transcript
        # decoded phrase as a list of words
        # emission_mx has a batch dimension as 1st dim (batch = number of audio files).
        transcripts = []
        for i in range(emission_mx.size(dim=0)):
            hyp_dict = dict()
            result = self.decoder([emission_mx[i]])
            transcript = ' '.join(result).lower()
            hyp_dict['pred_txt'] = transcript
            # add a hypothesis dict
            transcripts.append(hyp_dict)

        return transcripts


class BeamSearchDecoder_Torch(BaseDecoder):
    """Lexicon-free beam search decoder from torchaudio implementation without language model."""
    def __init__(self, vocab_path_or_bundle: str, num_hyps: int = 1, time_aligns: bool = False) -> None:
        self.num_hyps = num_hyps
        self.time_aligns = time_aligns
        # vocab is passed to the decoder object during initialisation.
        vocab = self._get_vocab(vocab_path_or_bundle)

        # the vocab is either taken from the torchaudio KenLM language model implementation or from a txt file
        # True if using the decoder in combination with a wav2vec2 model from a checkpoint file (need to use the same vocab for decoder and ASR model)
        vocab_from_txt = True if vocab_path_or_bundle.endswith('txt') else False
        # <s> is taken from fairseq.data.dictionary.Dictionary.bos_word, else use default value of '-'
        blank_token = '<s>' if vocab_from_txt else '-'

        # beam search decoder params.
        beam_size = 50 # default value in torchaudio.
        beam_threshold = 50 # default value in torchaudio.

        # initialise a beam search decoder.
        self.decoder = torchaudio.models.decoder.ctc_decoder(
            lexicon=None,
            lm=None,
            tokens=vocab, # same tokens as ASR model's that were used during inference.
            nbest=self.num_hyps, # number of top beams (hypotheses) to return.
            beam_size=beam_size,
            blank_token=blank_token,
            beam_threshold=beam_threshold
        )
        # print decoder info.
        logging.info("------ Decoder ------")
        logging.info("Lexicon-free beam search decoder without external language model, torchaudio implementation.")
        logging.info(f"Type: {self.decoder}")
        logging.info(f"Vocab: {vocab}")
        logging.info(f"Beam size: {beam_size}")
        logging.info(f"Returns best hypotheses: {self.decoder.nbest}")
        logging.info("---------------------")

    def generate(self, emission_mx: torch.Tensor, audio_lens: Optional[List[int]] = None) -> Union[List[Dict], List[List[Dict]]]:
        # emission_mx has a batch dimension as 1st dim (batch = number of audio files).
        hypos = self.decoder(emission_mx.cpu().detach())

        return beam_search_decode_torch(self.decoder, hypos, emission_mx, audio_lens, self.num_hyps, self.time_aligns)
    

class BeamSearchKenLMDecoder_Torch(BaseDecoder):
    """Lexicon-based beam search decoder from torchaudio implementation with a KenLM LibriSpeech 4-gram language model (lexicon from the LM, vocab from ASR acoustic model)."""
    def __init__(self, vocab_path_or_bundle: str, num_hyps: int = 1, time_aligns: bool = False) -> None:
        self.num_hyps = num_hyps
        self.time_aligns = time_aligns
        # vocab is passed to the decoder object during initialisation
        vocab = self._get_vocab(vocab_path_or_bundle)

        # get KenLM language model config
        files = torchaudio.models.decoder.download_pretrained_files("librispeech-4-gram")

        # the vocab is either taken from the torchaudio KenLM language model implementation or from a txt file
        # True if using the decoder in combination with a wav2vec2 model from a checkpoint file (need to use the same vocab for decoder and ASR model)
        vocab_from_txt = True if vocab_path_or_bundle.endswith('txt') else False
        # <s> is taken from fairseq.data.dictionary.Dictionary.bos_word, else use default value of '-'
        blank_token = '<s>' if vocab_from_txt else '-'

        # beam search decoder params.
        beam_size = 1500 # default value in torchaudio.
        lm_weight=3.23
        word_score=-0.26
        beam_threshold = 50

        # initialise a beam search decoder with a KenLM language model.
        self.decoder = torchaudio.models.decoder.ctc_decoder(
            lexicon=files.lexicon, # giant file of English "words".
            tokens=vocab, # same tokens as ASR model's that were used during inference.
            blank_token=blank_token, 
            lm=files.lm, # path to language model binary.
            nbest=self.num_hyps,
            beam_size=beam_size,
            beam_threshold=beam_threshold,
            lm_weight=lm_weight,
            word_score=word_score,
        )

        # print decoder info.
        logging.info("------ Decoder ------")
        logging.info("Lexicon-based beam search decoder from torchaudio with KenLM 4-gram language model.")
        logging.info(f"Type: {self.decoder}")
        logging.info(f"Vocab: {vocab}")
        logging.info(f"Returns best hypotheses: {self.decoder.nbest}")
        logging.info(f"Beam size: {beam_size}")
        logging.info(f"Beam threshold: {beam_threshold}")
        logging.info(f"Lexicon filepath: {files.lexicon}")
        logging.info(f"Language model filepath: {files.lm}")
        logging.info(f"Language model weight: {lm_weight}")
        logging.info(f"Lexicon word score: {word_score}")
        logging.info("---------------------")

    def generate(self, emission_mx: torch.Tensor, audio_lens: Optional[List[int]] = None) -> Union[List[Dict], List[List[Dict]]]:
        # emission_mx has a batch dimension as 1st dim (batch = number of audio files).
        hypos = self.decoder(emission_mx.cpu().detach())
        
        return beam_search_decode_torch(self.decoder, hypos, emission_mx, audio_lens, self.num_hyps, self.time_aligns)



class ViterbiDecoder(BaseDecoder):
    """Viterbi decoder from fairseq implementation.
    Theoretically, the Viterbi algorithm is not a greedy algorithm.
    It performs a global optimisation and guarantees to find the most likely state sequence, by exploring all possible state sequences.
    It does not use a language model as a postprocessing step for decoding the transcript.
    It CAN return multiple best hypotheses.

    However, in the fairseq library 'Viterbi' is used as the name of their greedy decoder, 
    which simply finds the most likely token at each timestep without context, using only acoustic model predictions.
    The fairseq Viterbi decoder can only return the best hypothesis.

    To be used with wav2vec2 acoustic models checkpoint files that were trained in the fairseq framework.
    """
    def __init__(self, vocab_path_or_bundle: str, num_hyps: bool = 1, time_aligns: bool = False) -> None:
        assert re.match(r'torchaudio.pipelines', vocab_path_or_bundle) is None, "Cannot provide a torch bundle to ViterbiDecoder, must use a txt file as vocab path."

        self.num_hyps = num_hyps
        self.time_aligns = time_aligns

        target_dict = Dictionary.load(vocab_path_or_bundle)
        # specify the number of best predictions to keep
        decoder_args = Namespace(**{'nbest': self.num_hyps})
        # vocab is passed to the decoder object during initialisation
        self.decoder = W2lViterbiDecoder(decoder_args, target_dict)

        # print decoder info.
        logging.info("------ Decoder ------")
        logging.info("Viterbi (greedy) Decoder, fairseq implementation.")
        logging.info(f"Type: {self.decoder}")
        logging.info(f"Vocab: {target_dict.symbols}")
        logging.info("Returns best hypotheses: 1")
        logging.info("---------------------")

    def generate(self, emission_mx: torch.Tensor, audio_lens: Optional[List[int]] = None) -> List[Dict]:
        # emission_mx has a batch dimension as 1st dim (batch = number of audio files).
        hypos = self.decoder.decode(emission_mx)
        transcripts = []
        for i in range(emission_mx.size(dim=0)):
            # if the batch_size is > 1, use the maximum original audio length in the batch, as all other audio files are padded to the max length during preprocessing.
            audio_len = audio_lens[i] if emission_mx.size(dim=0) == 1 else max(audio_lens)
            hyp_dict = dict()
            # append the decoded phrase (as a list of words) from the prediction of the first beam [0] (most likely transcript).
            transcript = ' '.join(hypos[i][0]['words']).lower()
            hyp_dict['pred_txt'] = transcript
            if self.time_aligns:
                word_times = get_word_time_alignments_fairseq(audio_len, emission_mx.size(dim=1), 16000, hypos[i][0]['symbols'], hypos[i][0]['timesteps'])
                timestamps_word = normalize_timestamp_output_w2v2(hyp_dict['pred_txt'].split(' '), word_times)
                hyp_dict['timestamps_word'] = timestamps_word
            # add a hypothesis dict
            transcripts.append(hyp_dict)

        return transcripts
    

class BeamSearchDecoder_Fairseq(BaseDecoder):
    """Lexicon-free beam search decoder without a language model from fairseq implementation.
    To be used with wav2vec2 acoustic models checkpoint files that were trained in the fairseq framework.
    """
    def __init__(self, vocab_path_or_bundle: str, num_hyps: int = 1, time_aligns: bool = False) -> None:
        assert re.match(r'torchaudio.pipelines', vocab_path_or_bundle) is None, "Cannot provide a torch bundle to BeamSearchDecoder_Fairseq, must use a txt file as vocab path."

        self.num_hyps = num_hyps
        self.time_aligns = time_aligns

        target_dict = Dictionary.load(vocab_path_or_bundle)

        # beam search decoder params.
        beam_size = 500
        beam_threshold = 25.0
        lexicon = ''
        # zero LM workaround - must provide a valid path to a language model, but set the weights of the LM to zero.
        lm_model = "/workspace/speech-augmentation/Models/language_models/lm_librispeech_kenlm_word_4g_200kvocab.bin"
        lm_weight = 0.0
        word_score = 0.0

        # specify non-default decoder arguments as a dict that is then converted to a Namespace object
        decoder_args = {
            'kenlm_model': lm_model, # path to KenLM binary, found at https://github.com/flashlight/wav2letter/tree/main/recipes/sota/2019#pre-trained-language-models
            'unit_lm': True,
            # used for both lexicon-based and lexicon-free beam search decoders
            'nbest': self.num_hyps, # number of best hypotheses to keep, a property of parent class (W2lDecoder)
            
            # see ref: https://github.com/flashlight/flashlight/blob/main/flashlight/app/asr/README.md#2-beam-search-optimization
            'beam': beam_size, # beam length (how many top tokens to keep at each timestep and therefore how many top hypotheses to generate after decoding all timesteps)
            'beam_threshold': beam_threshold, # at each timestep, tokens whose score gaps from the highest scored token are larger than the threshold are discarded

            'lm_weight': lm_weight, # how much the LM scores affect the hypotheses' scores
            'sil_weight': 0.0, # the silence token's weight
            # lexicon-based specific
            'lexicon': lexicon, # https://dl.fbaipublicfiles.com/textless_nlp/gslm/eval_data/lexicon_ltr.lst
            'word_score': word_score,
            'unk_weight': float('-inf'), # the unknown token's weight
        }
        decoder_args = Namespace(**decoder_args)
        # vocab is passed to the decoder object during initialisation
        self.decoder = W2lKenLMDecoder(decoder_args, target_dict)

        # print decoder info.
        logging.info("------ Decoder ------")
        logging.info("Lexicon-free beam search decoder without a language model, fairseq implementation.")
        logging.info(f"Type: {self.decoder}")
        logging.info(f"Vocab: {target_dict.symbols}")
        logging.info(f"Returns best hypotheses: {self.decoder.nbest}")
        logging.info(f"Beam size: {beam_size}")
        logging.info(f"Beam threshold: {beam_threshold}")
        logging.info("---------------------")

    def generate(self, emission_mx: torch.Tensor, audio_lens: Optional[List[int]] = None) -> Union[List[Dict], List[List[Dict]]]:
        # emission_mx has a batch dimension as 1st dim (batch = number of audio files).
        hypos = self.decoder.decode(emission_mx)
        
        return beam_search_decode_fairseq(hypos, emission_mx, audio_lens, self.num_hyps, self.time_aligns)


class BeamSearchKenLMDecoder_Fairseq(BaseDecoder):
    """Lexicon-based beam search decoder with a KenLM 4-gram language model from fairseq implementation.
    More consistent to be used with wav2vec2 acoustic models checkpoint files that were trained in the fairseq framework.
    """
    def __init__(self, vocab_path_or_bundle: str, num_hyps: int = 1, time_aligns: bool = False) -> None:
        assert re.match(r'torchaudio.pipelines', vocab_path_or_bundle) is None, "Cannot provide a torch bundle to BeamSearchKenLMDecoder_Fairseq, must use a txt file as vocab path."

        self.num_hyps = num_hyps
        self.time_aligns = time_aligns

        target_dict = Dictionary.load(vocab_path_or_bundle)

        # beam search decoder params.
        beam_size = 500
        beam_threshold = 25.0
        lm_model = "/workspace/speech-augmentation/Models/language_models/lm_librispeech_kenlm_word_4g_200kvocab.bin"
        lexicon = "/workspace/speech-augmentation/Models/language_models/lexicon_ltr.lst"
        word_score = -1.0
        lm_weight = 2.0

        # specify non-default decoder arguments as a dict that is then converted to a Namespace object
        decoder_args = {
            'kenlm_model': lm_model, # path to KenLM binary, found at https://github.com/flashlight/wav2letter/tree/main/recipes/sota/2019#pre-trained-language-models
            # used for both lexicon-based and lexicon-free beam search decoders
            'nbest': self.num_hyps, # number of best hypotheses to keep, a property of parent class (W2lDecoder)
            
            # see ref: https://github.com/flashlight/flashlight/blob/main/flashlight/app/asr/README.md#2-beam-search-optimization
            'beam': beam_size, # beam length (how many top tokens to keep at each timestep and therefore how many top hypotheses to generate after decoding all timesteps)
            'beam_threshold': beam_threshold, # at each timestep, tokens whose score gaps from the highest scored token are larger than the threshold are discarded

            'lm_weight': lm_weight, # how much the LM scores affect the hypotheses' scores
            'sil_weight': 0.0, # the silence token's weight
            # lexicon-based specific
            'lexicon': lexicon, # https://dl.fbaipublicfiles.com/textless_nlp/gslm/eval_data/lexicon_ltr.lst
            'word_score': word_score,
            'unk_weight': float('-inf'), # the unknown token's weight
        }
        decoder_args = Namespace(**decoder_args)
        # vocab is passed to the decoder object during initialisation
        self.decoder = W2lKenLMDecoder(decoder_args, target_dict)

        # print decoder info.
        logging.info("------ Decoder ------")
        logging.info("Lexicon-based beam search decoder with a KenLM 4-gram language model from fairseq.")
        logging.info(f"Type: {self.decoder}")
        logging.info(f"Vocab: {target_dict.symbols}")
        logging.info(f"Returns best hypotheses: {self.decoder.nbest}")
        logging.info(f"Beam size: {beam_size}")
        logging.info(f"Beam threshold: {beam_threshold}")
        logging.info(f"Lexicon filepath: {lexicon}")
        logging.info(f"Language model filepath: {lm_model}")
        logging.info(f"Language model weight: {lm_weight}")
        logging.info(f"Lexicon word score: {word_score}")
        logging.info("---------------------")

    def generate(self, emission_mx: torch.Tensor, audio_lens: Optional[List[int]] = None) -> Union[List[Dict], List[List[Dict]]]:
        # emission_mx has a batch dimension as 1st dim (batch = number of audio files).
        hypos = self.decoder.decode(emission_mx)
        
        return beam_search_decode_fairseq(hypos, emission_mx, audio_lens, self.num_hyps, self.time_aligns)


class TransformerDecoder(BaseDecoder):
    """Lexicon-based beam search decoder with a Transformer language model from fairseq implementation.
    The pretrained fairseq Transformer language model mentioned in the original wav2vec2 paper is used (Librispeech).
    """
    def __init__(self, vocab_path_or_bundle: str, num_hyps: int = 1, time_aligns: bool = False) -> None:
        assert re.match(r'torchaudio.pipelines', vocab_path_or_bundle) is None, "Cannot provide a torch bundle to TransformerDecoder, must use a txt file as vocab path."

        self.num_hyps = num_hyps
        self.time_aligns = time_aligns

        target_dict = Dictionary.load(vocab_path_or_bundle) # path to the freq table of chars used to finetune the wav2vec2 model.
        # Path to folder that contains the trained TransformerLM binary.
        # Download the .pt file from https://github.com/flashlight/wav2letter/tree/main/recipes/sota/2019#pre-trained-language-models
        # NOTE: there must also be a 'dict.txt' file in the folder.
        # This is the freq table of words used to train the LM (usually trained on Librispeech). 
        # Download the TransformerLM dict file (called 'lm_librispeech_word_transformer.dict') from the same link as above and rename it to 'dict.txt'.
        # Make sure all characters are upper cased!
        transformerLM_root_folder = "/workspace/speech-augmentation/Models/language_models/transformer_lm"
        convert_to_upper(os.path.join(transformerLM_root_folder, 'dict.txt'))
        # specify non-default decoder arguments as a dict that is then converted to a Namespace object

        # beam search decoder params.
        beam_size = 500
        beam_threshold = 25.0
        lm_model = os.path.join(transformerLM_root_folder, "lm_librispeech_word_transformer.pt")
        lexicon = "/workspace/speech-augmentation/Models/language_models/lexicon_ltr.lst"
        word_score = -1.0
        lm_weight = 2.0


        decoder_args = {
            'kenlm_model': lm_model,
            # used for both lexicon-based and lexicon-free beam search decoders
            'nbest': self.num_hyps, # number of best hypotheses to keep, a property of parent class (W2lDecoder)
            'beam': beam_size, # beam length
            'beam_threshold': beam_threshold,
            'lm_weight': lm_weight,
            'sil_weight': 0.0, # the silence token's weight
            # lexicon-based specific
            'lexicon': lexicon, # https://dl.fbaipublicfiles.com/textless_nlp/gslm/eval_data/lexicon_ltr.lst
            'word_score': word_score,
            'unk_weight': float('-inf'), # the unknown token's weight
        }
        decoder_args = Namespace(**decoder_args)
        # vocab is passed to the decoder object during initialisation
        self.decoder = W2lFairseqLMDecoder(decoder_args, target_dict)

        # print decoder info.
        logging.info("------ Decoder ------")
        logging.info("Lexicon-based beam search decoder with a Transformer language model from fairseq.")
        logging.info(f"Type: {self.decoder}")
        logging.info(f"Vocab: {self.decoder.symbols}")
        logging.info(f"Returns best hypotheses: {self.decoder.nbest}")
        logging.info(f"Beam size: {beam_size}")
        logging.info(f"Beam threshold: {beam_threshold}")
        logging.info(f"Lexicon filepath: {lexicon}")
        logging.info(f"Language model filepath: {lm_model}")
        logging.info(f"Language model weight: {lm_weight}")
        logging.info(f"Lexicon word score: {word_score}")
        logging.info("---------------------")

    def generate(self, emission_mx: torch.Tensor, audio_lens: Optional[List[int]] = None) -> Union[List[Dict], List[List[Dict]]]:
        # emission_mx has a batch dimension as 1st dim (batch = number of audio files).
        hypos = self.decoder.decode(emission_mx)

        return beam_search_decode_fairseq(hypos, emission_mx, audio_lens, self.num_hyps, self.time_aligns)



class ASR_Decoder_Pair():
    """A bundle representing a particular combination of an ASR model and decoder.
    The combination is stored in this single object to prevent using an ASR model and decoder that are incompatible or that were initialised with different vocabs.
    """

    def __init__(self, model: BaseWav2Vec2Model, decoder: BaseDecoder) -> None:
        self.model = model
        self.decoder = decoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _infer_sequential(self, filepaths: List[str]) -> Union[Union[List[str], List[Tuple[str, Tuple[float, float]]]], List[List[Tuple[str, Tuple[float, float]]]]]:
        """Performs sequential inference on a list of audio filepath(s), processed one at a time, using a particular combination of ASR model and decoder."""

        transcripts = []
        for filepath in tqdm(filepaths, total=len(filepaths), unit=" transcript", desc="Generating transcripts predictions sequentially, so far"):
            logging.info(f"Generating transcript for {filepath}")
            # emission_mx is a batch of emission matrices, one per audio file, but here batch dimension is always of size 1.
            emission_mx, audio_len = self.model.forward([filepath], self.device)
            transcript = self.decoder.generate(emission_mx, audio_len)[0] # batch_size = 1, so only [0] element present
            transcripts.append(transcript)
            logging.info(f"Transcript for {filepath} generated.")

        return transcripts

    def infer(self, filepaths: List[str], batch_size: int=1) -> Union[List[Dict], List[List[Dict]]]:
        """Performs minibatched inference on a list of audio sample(s) specified by the filepath(s), using a particular combination of ASR model and decoder.
        If batch_size=1, performs sequential inference where audio samples are passed through the ASR model one at a time.
        If batch_size=len(filepaths), performs batched inference where all audio samples are processed at one time as a single input matrix.
        If 0 < batch_size < len(filepaths), performs minibatch inference.
        Only ASR inference is batched, while audio preprocessing and decoding is done sequentially, sample by sample.
        
        Returns:
            transcripts (Union[List[Dict], List[List[Dict]]])

            'batch_size' does not affect whether a list or a list of lists is returned as 'transcripts'.
            The top list will always have one element per audio file, regardless if the all the audio data was processed sequentially or in a batch/minibatch.
            if args.num_hyps=1: The elements of the 'transcripts' list will be hypothesis dict objects (just the best hypothesis per audio file).
            if args.num_hyps>1, and the decoder supports it: The elements of the top list will be lists, where the list elements will be comprised of args.num_hyps number of hypothesis dict objects.
        """
        # initialise the transcripts list for all files
        transcripts = []


        if batch_size == 1:
            # sequential inference
            transcripts = self._infer_sequential(filepaths)
        else:
            # minibatch inference
            assert batch_size <= len(filepaths), "ERROR: batch_size must be less than or equal to the number of audio samples to process for inference."
            for i in tqdm(range(0, len(filepaths), batch_size), total=int(math.ceil(len(filepaths)/batch_size)), unit=" minibatch", desc="Generating transcripts in minibatches, so far"):
                logging.info(f"Generating transcripts for batch of wavs of size {batch_size}")
                # emission_mx is a batch of emission matrices (one 2D matrix per audio file of size 'num_frames' * 'ASR output chars').
                emission_mx, audio_lens = self.model.forward(filepaths[i:i+batch_size], self.device)
                transcripts.append(self.decoder.generate(emission_mx, audio_lens))
                logging.info(f"Transcripts for batch generated.")

            # flatten the minibatch lists
            transcripts = [transcript for minilist in transcripts for transcript in minilist]

        return transcripts


class Wav2Vec2_Decoder_Factory():
    """ASR factory class - used to return a particular combination of an ASR acoustic model and an ASR decoder.
    No need to create an instance object of the factory.
    Defines a class method for each combination of wav2vec2 model and decoder.
    
    Each class method returns a wrapped tuple: ASR_Decoder_Pair((BaseWav2Vec2Model, BaseDecoder))
        - wav2vec2 ASR model object is of type BaseWav2Vec2Model to keep the interface consistent, regardless of how the model was loaded.
        - ASR decoder object is of type BaseDecoder to keep the interface consistent, regardless of what decoder is used.
        - these two objects are then wrapped in a wrapper class so that these two objects are always associated together and not used separately.
        - this ensures that the decoder has the same vocab list as the ASR model for correct token decoding of the model output.
    """

    @classmethod
    def get_torchaudio_greedy(cls, bundle_str: str='torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H', num_hyps: Optional[int]=None, time_aligns: Optional[bool]=None) -> ASR_Decoder_Pair:
        """returns a torchaudio wav2vec2 model and a greedy decoder from torchaudio."""
        # using largest available wav2vec2 model from torchaudio by default
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(TorchaudioWav2Vec2Model(device=device, vocab_path_or_bundle=bundle_str),
                                GreedyDecoder(vocab_path_or_bundle=bundle_str))

    @classmethod
    def get_torchaudio_beamsearch(cls, bundle_str: str='torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H', num_hyps: int=1, time_aligns: bool=False) -> ASR_Decoder_Pair:
        """returns a torchaudio wav2vec2 model and a lexicon-free beam search decoder from torchaudio without an external language model."""
        # using largest available wav2vec2 model from torchaudio
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(TorchaudioWav2Vec2Model(device=device, vocab_path_or_bundle=bundle_str),
                                BeamSearchDecoder_Torch(vocab_path_or_bundle=bundle_str, num_hyps=num_hyps, time_aligns=time_aligns))

    @classmethod
    def get_torchaudio_beamsearchkenlm(cls, bundle_str: str='torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H', num_hyps: int=1, time_aligns: bool=False) -> ASR_Decoder_Pair:
        """returns a torchaudio wav2vec2 model and a lexicon-based beam search decoder from torchaudio coupled with a KenLM external language model."""
        # using largest available wav2vec2 model from torchaudio
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(TorchaudioWav2Vec2Model(device=device, vocab_path_or_bundle=bundle_str),
                                BeamSearchKenLMDecoder_Torch(vocab_path_or_bundle=bundle_str, num_hyps=num_hyps, time_aligns=time_aligns))

    @classmethod
    def get_args_viterbi(cls, model_filepath: str, vocab_path: str, num_hyps: Optional[int]=None, time_aligns: bool=False) -> ASR_Decoder_Pair:
        """returns a wav2vec2 model loaded from a custom checkpoint (trained in the fairseq framework) that has an args field and a Viterbi decoder from fairseq."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(ArgsWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                ViterbiDecoder(vocab_path_or_bundle=vocab_path, num_hyps=num_hyps, time_aligns=time_aligns))

    @classmethod
    def get_cfg_viterbi(cls, model_filepath: str, vocab_path: str, num_hyps: Optional[int]=None, time_aligns: bool=False) -> ASR_Decoder_Pair:
        """returns a wav2vec2 model loaded from a custom checkpoint (trained in the fairseq framework) that has a cfg field and a Viterbi decoder from fairseq."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(CfgWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                ViterbiDecoder(vocab_path_or_bundle=vocab_path, num_hyps=num_hyps, time_aligns=time_aligns))

    @classmethod
    def get_args_greedy(cls, model_filepath: str, vocab_path: str, num_hyps: Optional[int]=None, time_aligns: Optional[bool]=None) -> ASR_Decoder_Pair:
        """returns a wav2vec2 model loaded from a custom checkpoint (trained in the fairseq framework) that has an args field and a greedy decoder from torchaudio."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(ArgsWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                GreedyDecoder(vocab_path_or_bundle=vocab_path))

    @classmethod
    def get_cfg_greedy(cls, model_filepath: str, vocab_path: str, num_hyps: Optional[int]=None, time_aligns: Optional[bool]=None) -> ASR_Decoder_Pair:
        """returns a wav2vec2 model loaded from a custom checkpoint (trained in the fairseq framework) that has a cfg field and a greedy decoder from torchaudio."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(CfgWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                GreedyDecoder(vocab_path_or_bundle=vocab_path))

    @classmethod
    def get_args_beamsearch_torch(cls, model_filepath: str, vocab_path: str, num_hyps: int=1, time_aligns: bool=False) -> ASR_Decoder_Pair:
        """returns a wav2vec2 model loaded from a custom checkpoint (trained in the fairseq framework) that has an args field and a beam search decoder from torchaudio without an external language model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(ArgsWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                BeamSearchDecoder_Torch(vocab_path_or_bundle=vocab_path, num_hyps=num_hyps, time_aligns=time_aligns))

    @classmethod
    def get_cfg_beamsearch_torch(cls, model_filepath: str, vocab_path: str, num_hyps: int=1, time_aligns: bool=False) -> ASR_Decoder_Pair:
        """returns a wav2vec2 model loaded from a custom checkpoint (trained in the fairseq framework) that has a cfg field and a beam search decoder from torchaudio without an external language model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(CfgWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                BeamSearchDecoder_Torch(vocab_path_or_bundle=vocab_path, num_hyps=num_hyps, time_aligns=time_aligns))

    # 
    def get_args_beamsearchkenlm_torch(cls, model_filepath: str, vocab_path: str, num_hyps: int=1, time_aligns: bool=False) -> ASR_Decoder_Pair:
        """returns a wav2vec2 model loaded from a custom checkpoint (trained in the fairseq framework) that has an args field and a beam search decoder from torchaudio with a KenLM external language model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(ArgsWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                BeamSearchKenLMDecoder_Torch(vocab_path_or_bundle=vocab_path, num_hyps=num_hyps, time_aligns=time_aligns))

    @classmethod
    def get_cfg_beamsearchkenlm_torch(cls, model_filepath: str, vocab_path: str, num_hyps: int=1, time_aligns: bool=False) -> ASR_Decoder_Pair:
        """returns a wav2vec2 model loaded from a custom checkpoint (trained in the fairseq framework) that has a cfg field and a beam search decoder from torchaudio with a KenLM external language model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(CfgWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                BeamSearchKenLMDecoder_Torch(vocab_path_or_bundle=vocab_path, num_hyps=num_hyps, time_aligns=time_aligns))
    
    @classmethod
    def get_args_beamsearch_fairseq(cls, model_filepath: str, vocab_path: str, num_hyps: int=1, time_aligns: bool=False) -> ASR_Decoder_Pair:
        """returns a wav2vec2 model loaded from a custom checkpoint (trained in the fairseq framework) that has an args field and a beam search decoder from fairseq without an external language model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(ArgsWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                BeamSearchDecoder_Fairseq(vocab_path_or_bundle=vocab_path, num_hyps=num_hyps, time_aligns=time_aligns))

    @classmethod
    def get_cfg_beamsearch_fairseq(cls, model_filepath: str, vocab_path: str, num_hyps: int=1, time_aligns: bool=False) -> ASR_Decoder_Pair:
        """returns a wav2vec2 model loaded from a custom checkpoint (trained in the fairseq framework) that has a cfg field and a beam search decoder from fairseq without an external language model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(CfgWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                BeamSearchDecoder_Fairseq(vocab_path_or_bundle=vocab_path, num_hyps=num_hyps, time_aligns=time_aligns))

    @classmethod
    def get_args_beamsearchkenlm_fairseq(cls, model_filepath: str, vocab_path: str, num_hyps: int=1, time_aligns: bool=False) -> ASR_Decoder_Pair:
        """returns a wav2vec2 model loaded from a custom checkpoint (trained in the fairseq framework) that has an args field and a beam search decoder from fairseq with a KenLM external language model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(ArgsWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                BeamSearchKenLMDecoder_Fairseq(vocab_path_or_bundle=vocab_path, num_hyps=num_hyps, time_aligns=time_aligns))

    @classmethod
    def get_cfg_beamsearchkenlm_fairseq(cls, model_filepath: str, vocab_path: str, num_hyps: int=1, time_aligns: bool=False) -> ASR_Decoder_Pair:
        """returns a wav2vec2 model loaded from a custom checkpoint (trained in the fairseq framework) that has a cfg field and a beam search decoder from fairseq with a KenLM external language model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(CfgWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                BeamSearchKenLMDecoder_Fairseq(vocab_path_or_bundle=vocab_path, num_hyps=num_hyps, time_aligns=time_aligns))

    @classmethod
    def get_args_beamsearchtransformerlm(cls, model_filepath: str, vocab_path: str, num_hyps: int=1, time_aligns: bool=False) -> ASR_Decoder_Pair:
        """returns a wav2vec2 model loaded from a custom checkpoint (trained in the fairseq framework) that has an args field and a beam search decoder from fairseq with a Transformer external language model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(ArgsWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                TransformerDecoder(vocab_path_or_bundle=vocab_path, num_hyps=num_hyps, time_aligns=time_aligns))

    @classmethod
    def get_cfg_beamsearchtransformerlm(cls, model_filepath: str, vocab_path: str, num_hyps: int=1, time_aligns: bool=False) -> ASR_Decoder_Pair:
        """returns a wav2vec2 model loaded from a custom checkpoint (trained in the fairseq framework) that has a cfg field and a beam search decoder from fairseq with a Transformer external language model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(CfgWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                TransformerDecoder(vocab_path_or_bundle=vocab_path, num_hyps=num_hyps, time_aligns=time_aligns))


def main() -> None:
    # model configs
    bundle_str = "torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H"
    # wav2vec2 models checkpoints that were trained in the fairseq framework
    args_model_filepath = "/workspace/projects/speech-augmentation/Models/w2v_fairseq/wav2vec2_vox_960h_new.pt"
    cfg_model_filepath = "/workspace/projects/speech-augmentation/Models/vox_55h/checkpoints/checkpoint_best.pt"
    # vocab dicts used during the training of the corresponding wav2vec2 models trained in the fairseq framework
    args_vocab_filepath = "/workspace/projects/speech-augmentation/Models/w2v_fairseq/dict.ltr.txt"
    cfg_vocab_filepath = "/workspace/projects/speech-augmentation/Models/vox_55h/dict.ltr.txt"
    # audio samples to test inference on
    wavpaths = ["/workspace/datasets/myst_test/myst_999465_2009-17-12_00-00-00_MS_4.2_024.wav",
                "/workspace/datasets/myst_test/myst_002030_2014-02-28_09-37-51_LS_1.1_006.wav",
                "/workspace/datasets/myst_valid/002013/myst_002013_2014-03-11_11-14-16_LS_2.1_042.wav",
                "/workspace/datasets/myst_valid/002013/myst_002013_2014-03-11_11-14-16_LS_2.1_043.wav",
                "/workspace/datasets/myst_valid/002013/myst_002013_2014-03-11_11-14-16_LS_2.1_045.wav",
                "/workspace/datasets/myst_valid/002013/myst_002013_2014-03-11_11-14-16_LS_2.1_048.wav",
                "/workspace/datasets/myst_valid/002013/myst_002013_2014-03-11_11-14-16_LS_2.1_051.wav",
                ]

    # # test torchaudio wav2vec2 + greedy decoder -> works
    # asr1 = Wav2Vec2_Decoder_Factory.get_torchaudio_greedy(bundle_str=bundle_str)
    # transcripts1 = asr1.infer(wavpaths, batch_size=3)

    # # test torchaudio wav2vec2 + beam search decoder with KenLM language model from torchaudio -> works
    # asr2 = Wav2Vec2_Decoder_Factory.get_torchaudio_beamsearchkenlm(bundle_str=bundle_str)
    # transcripts2 = asr2.infer(wavpaths, batch_size=3)

    # # test args wav2vec2 + viterbi from fairseq -> works
    # asr3 = Wav2Vec2_Decoder_Factory.get_args_viterbi(model_filepath=args_model_filepath, vocab_path=args_vocab_filepath)
    # transcripts3 = asr3.infer(wavpaths, batch_size=3)

    # test cfg wav2vec2 + viterbi from fairseq -> works
    asr4 = Wav2Vec2_Decoder_Factory.get_cfg_viterbi(model_filepath=cfg_model_filepath, vocab_path=cfg_vocab_filepath)
    transcripts4 = asr4.infer(wavpaths, batch_size=3)

    # # test args wav2vec2 + greedy decoder -> works
    # asr5 = Wav2Vec2_Decoder_Factory.get_args_greedy(model_filepath=args_model_filepath, vocab_path=args_vocab_filepath)
    # transcripts5 = asr5.infer(wavpaths, batch_size=3)

    # # test cfg wav2vec2 + greedy decoder -> works
    # asr6 = Wav2Vec2_Decoder_Factory.get_cfg_greedy(model_filepath=cfg_model_filepath, vocab_path=cfg_vocab_filepath)
    # transcripts6 = asr6.infer(wavpaths, batch_size=3)

    # # test args wav2vec2 + beam search decoder with KenLM language model from torchaudio -> works
    # asr7 = Wav2Vec2_Decoder_Factory.get_args_beamsearchkenlm_torch(model_filepath=args_model_filepath, vocab_path=args_vocab_filepath)
    # transcripts7 = asr7.infer(wavpaths, batch_size=3)

    # # test cfg wav2vec2 + beam search decoder with KenLM language model from torchaudio-> works
    # asr8 = Wav2Vec2_Decoder_Factory.get_cfg_beamsearchkenlm_torch(model_filepath=cfg_model_filepath, vocab_path=cfg_vocab_filepath)
    # transcripts8 = asr8.infer(wavpaths, batch_size=3)

    # # test args wav2vec2 + beam search decoder with KenLM language model from fairseq-> works
    # asr9 = Wav2Vec2_Decoder_Factory.get_args_beamsearchkenlm_fairseq(model_filepath=args_model_filepath, vocab_path=args_vocab_filepath)
    # transcripts9 = asr9.infer(wavpaths, batch_size=3)

    # # test cfg wav2vec2 + beam search decoder with KenLM language model from fairseq -> works
    # asr10 = Wav2Vec2_Decoder_Factory.get_cfg_beamsearchkenlm_fairseq(model_filepath=cfg_model_filepath, vocab_path=cfg_vocab_filepath)
    # transcripts10 = asr10.infer(wavpaths, batch_size=3)

    # # test args wav2vec2 + beam search decoder with Transformer language model from fairseq -> works
    # asr11 = Wav2Vec2_Decoder_Factory.get_args_beamsearchtransformerlm(model_filepath=args_model_filepath, vocab_path=args_vocab_filepath)
    # transcripts11 = asr11.infer(wavpaths, batch_size=3)

    # # test cfg wav2vec2 + beam search decoder with Transformer language model from fairseq -> works
    # asr12 = Wav2Vec2_Decoder_Factory.get_cfg_beamsearchtransformerlm(model_filepath=cfg_model_filepath, vocab_path=cfg_vocab_filepath)
    # transcripts12 = asr12.infer(wavpaths, batch_size=3)


if __name__ == "__main__":
    main()