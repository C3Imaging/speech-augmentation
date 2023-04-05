# using principles of dependency injection and dependency inversion and creating factories to simplify API of using wav2vec2 models loaded in different ways + various decoders for ASR inference

import re
import os
import math
import torch
import fileinput
import torchaudio
from tqdm import tqdm
from typing import List, Any
from argparse import Namespace
from omegaconf import OmegaConf
from abc import ABC, abstractmethod
from fairseq.data import Dictionary
from fairseq.data.data_utils import post_process
if __name__ == "__main__":
    import decoding_utils_chkpt, decoding_utils_torch
else:
    from . import decoding_utils_chkpt, decoding_utils_torch
from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder, W2lKenLMDecoder, W2lFairseqLMDecoder
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc, Wav2Vec2CtcConfig


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
    def forward(self, filepaths: List[str], device: torch.device) -> torch.Tensor:
        """Runs inference on a batch of audio samples, first preprocessing the audio as required by the model and then returning the emissions matrices."""

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

    def forward(self, filepaths: List[str], device: torch.device) -> torch.Tensor:
        """can only perform single audio file inference."""
        waveforms = self._preproc(filepaths, self.sample_rate)
        assert len(waveforms) == 1, "ERROR!!! Length of waveforms list should be 1."
        # unpack single element of list
        waveform = waveforms[0].to(device)

        emissions, _ = self.model(waveform.unsqueeze(0)) # add a batch dimension

        return emissions


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

    def forward(self, filepaths: List[str], device: torch.device) -> torch.Tensor:
        """Inference is not called directly in the fairseq framework API but is done through a decoder, but I copy-pasted the actual inference part from
        fairseq/examples/speech_recognition/w2l_decoder.W2lDecoder.generate()"""
        waveforms = self._preproc(filepaths, self.sample_rate)
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

        return emissions


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

    def forward(self, filepaths: List[str], device: torch.device) -> torch.Tensor:
        """Inference is not called directly in the fairseq framework API but is done through a decoder, but I copy-pasted the actual inference part from
        fairseq/examples/speech_recognition/w2l_decoder.W2lDecoder.generate()"""
        waveforms = self._preproc(filepaths, self.sample_rate)
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

        return emissions


# ASR decoder abstract class
class BaseDecoder(ABC):
    # show what instance attributes should be defined
    decoder: Any # different decoders do not have a common interface

    @abstractmethod
    def generate(self, emission_mx: torch.Tensor) -> List[str]:
        """Generates a list of transcripts by decoding the output batch of a wav2vec2 ASR acoutic model."""

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
    """
    def __init__(self, vocab_path_or_bundle: str) -> None:
        # the vocabulary of chars known by the acoustic model that will be used for decoding
        # vocab is passed to the decoder object during initialisation
        vocab = self._get_vocab(vocab_path_or_bundle)
        self.decoder = decoding_utils_torch.GreedyCTCDecoder(vocab)

    def generate(self, emission_mx: torch.Tensor) -> List[str]:
        # generate one transcript
        # decoded phrase as a list of words
        # emission_mx should have a batch dimension
        transcripts = []
        for i in range(emission_mx.size(dim=0)):
            result = self.decoder([emission_mx[i]])
            transcript = ' '.join(result).lower()
            transcripts.append(transcript)

        return transcripts


class BeamSearchKenLMDecoder_Torch(BaseDecoder):
    """Lexicon-based beam search decoder with a KenLM 4-gram language model from torchaudio implementation."""
    def __init__(self, vocab_path_or_bundle: str) -> None:
        # vocab is passed to the decoder object during initialisation
        vocab = self._get_vocab(vocab_path_or_bundle)

        # get KenLM language model config
        files = torchaudio.models.decoder.download_pretrained_files("librispeech-4-gram")

        # the vocab is either taken from the torchaudio KenLM language model implementation or from a txt file
        # True if using the decoder in combination with a wav2vec2 model from a checkpoint file (need to use the same vocab for decoder and ASR model)
        vocab_from_txt = True if vocab_path_or_bundle.endswith('txt') else False
        # <pad> is taken from fairseq.data.dictionary.Dictionary.target_dict.pad_word, else use default value of '-'
        blank_token = '<pad>' if vocab_from_txt else '-'

        # initialise a beam search decoder with a KenLM language model
        self.decoder = torchaudio.models.decoder.ctc_decoder(
            lexicon=files.lexicon, # giant file of English "words"
            tokens=vocab, # same as wav2vec2's vocab
            blank_token=blank_token, 
            lm=files.lm, # path to language model binary
            nbest=3,
            beam_size=1500,
            lm_weight=3.23,
            word_score=-0.26,
        )

    def generate(self, emission_mx: torch.Tensor) -> List[str]:
        # emission_mx should have a batch dimension
        transcripts = []
        results = self.decoder(emission_mx.cpu().detach())
        for i in range(emission_mx.size(dim=0)):
            # select the decoded phrase as a list of words from the prediction of the first beam (most likely transcript)
            result = results[i][0].words
            transcript = ' '.join(result).lower()
            transcripts.append(transcript)

        return transcripts


class ViterbiDecoder(BaseDecoder):
    """Viterbi decoder from fairseq implementation.
    Theoretically, the Viterbi algorithm is not a greedy algorithm.
    It performs a global optimisation and guarantees to find the most likely state sequence, by exploring all possible state sequences.
    It does not use a language model as a postprocessing step for decoding the transcript.

    However, in the fairseq library 'Viterbi' is used as the name of their greedy decoder, 
    which simply finds the most likely token at each timestep without context, using only acoustic model predictions.
    """
    def __init__(self, vocab_path_or_bundle: str) -> None:
        assert re.match(r'torchaudio.pipelines', vocab_path_or_bundle) is None, "Cannot provide a torch bundle to ViterbiDecoder, must use a txt file."
        target_dict = Dictionary.load(vocab_path_or_bundle)
        # specify the number of best predictions to keep
        decoder_args = Namespace(**{'nbest': 1})
        # vocab is passed to the decoder object during initialisation
        self.decoder = W2lViterbiDecoder(decoder_args, target_dict)

    def generate(self, emission_mx: torch.Tensor) -> List[str]:
        # emission_mx should have a batch dimension
        hypos = self.decoder.decode(emission_mx) # add a batch dimension
        transcripts = [post_process(self.decoder.tgt_dict.string(h[0]['tokens'].int().cpu()), 'letter').lower() for h in hypos]

        return transcripts


class BeamSearchKenLMDecoder_Fairseq(BaseDecoder):
    """Lexicon-based beam search decoder with a KenLM 4-gram language model from fairseq implementation.
    More consistent to be used with wav2vec2 acoustic models checkpoint files that were trained in the fairseq framework.
    """
    def __init__(self, vocab_path_or_bundle: str) -> None:
        assert re.match(r'torchaudio.pipelines', vocab_path_or_bundle) is None, "Cannot provide a torch bundle to BeamSearchKenLMDecoder_Fairseq, must use a txt file."
        target_dict = Dictionary.load(vocab_path_or_bundle)
        # specify non-default decoder arguments as a dict that is then converted to a Namespace object
        decoder_args = {
            'kenlm_model': "/workspace/projects/Alignment/wav2vec2_alignment/Models/language_models/lm_librispeech_kenlm_word_4g_200kvocab.bin", # path to KenLM binary, found at https://github.com/flashlight/wav2letter/tree/main/recipes/sota/2019#pre-trained-language-models
            # used for both lexicon-based and lexicon-free beam search decoders
            'nbest': 1, # number of best hypotheses to keep, a property of parent class (W2lDecoder)
            
            # see ref: https://github.com/flashlight/flashlight/blob/main/flashlight/app/asr/README.md#2-beam-search-optimization
            'beam': 500, # beam length (how many top tokens to keep at each timestep and therefore how many top hypotheses to generate after decoding all timesteps)
            'beam_threshold': 25.0, # at each timestep, tokens whose score gaps from the highest scored token are larger than the threshold are discarded

            'lm_weight': 2.0, # how much the LM scores affect the hypotheses' scores
            'sil_weight': 0.0, # the silence token's weight
            # lexicon-based specific
            'lexicon': "/workspace/projects/Alignment/wav2vec2_alignment/Models/language_models/lexicon_ltr.lst", # https://dl.fbaipublicfiles.com/textless_nlp/gslm/eval_data/lexicon_ltr.lst
            'word_score': -1.0,
            'unk_weight': float('-inf'), # the unknown token's weight
        }
        decoder_args = Namespace(**decoder_args)
        # vocab is passed to the decoder object during initialisation
        self.decoder = W2lKenLMDecoder(decoder_args, target_dict)

    def generate(self, emission_mx: torch.Tensor) -> List[str]:
        # emission_mx should have a batch dimension
        hypos = self.decoder.decode(emission_mx) # add a batch dimension
        transcripts = [post_process(self.decoder.tgt_dict.string(h[0]['tokens'].int().cpu()), 'letter').lower() for h in hypos]

        return transcripts


class TransformerDecoder(BaseDecoder):
    """Lexicon-based beam search decoder with a Transformer language model from fairseq implementation.
    The pretrained fairseq Transformer language model mentioned in the original wav2vec2 paper is used (Librispeech).
    """
    def __init__(self, vocab_path_or_bundle: str) -> None:
            assert re.match(r'torchaudio.pipelines', vocab_path_or_bundle) is None, "Cannot provide a torch bundle to TransformerDecoder, must use a txt file."
            target_dict = Dictionary.load(vocab_path_or_bundle) # path to the freq table of chars used to finetune the wav2vec2 model.
            # Path to folder that contains the trained TransformerLM binary.
            # Download the .pt file from https://github.com/flashlight/wav2letter/tree/main/recipes/sota/2019#pre-trained-language-models
            # NOTE: there must also be a 'dict.txt' file in the folder.
            # This is the freq table of words used to train the LM (usually trained on Librispeech). 
            # Download the TransformerLM dict file (called 'lm_librispeech_word_transformer.dict') from the same link as above and rename it to 'dict.txt'.
            # Make sure all characters are upper cased!
            transformerLM_root_folder = "/workspace/projects/Alignment/wav2vec2_alignment/Models/language_models/transformer_lm"
            convert_to_upper(os.path.join(transformerLM_root_folder, 'dict.txt'))
            # specify non-default decoder arguments as a dict that is then converted to a Namespace object
            decoder_args = {
                'kenlm_model': os.path.join(transformerLM_root_folder, "lm_librispeech_word_transformer.pt"),
                # used for both lexicon-based and lexicon-free beam search decoders
                'nbest': 1, # number of best hypotheses to keep, a property of parent class (W2lDecoder)
                'beam': 500, # beam length
                'beam_threshold': 25.0,
                'lm_weight': 2.0,
                'sil_weight': 0.0, # the silence token's weight
                # lexicon-based specific
                'lexicon': "/workspace/projects/Alignment/wav2vec2_alignment/Models/language_models/lexicon_ltr.lst", # https://dl.fbaipublicfiles.com/textless_nlp/gslm/eval_data/lexicon_ltr.lst
                'word_score': -1.0,
                'unk_weight': float('-inf'), # the unknown token's weight
            }
            decoder_args = Namespace(**decoder_args)
            # vocab is passed to the decoder object during initialisation
            self.decoder = W2lFairseqLMDecoder(decoder_args, target_dict)

    def generate(self, emission_mx: torch.Tensor) -> List[str]:
        # emission_mx should have a batch dimension
        hypos = self.decoder.decode(emission_mx) # add a batch dimension
        transcripts = [post_process(self.decoder.tgt_dict.string(h[0]['tokens'].int().cpu()), 'letter').lower() for h in hypos]

        return transcripts


class ASR_Decoder_Pair():
    """A bundle representing a particular combination of an ASR model and decoder.
    The combination is stored in this single object to prevent using an ASR model and decoder that are incompatible or that were initialised with different vocabs.
    """

    def __init__(self, model: BaseWav2Vec2Model, decoder: BaseDecoder) -> None:
        self.model = model
        self.decoder = decoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _infer_sequential(self, filepaths: List[str]) -> List[str]:
        """Performs sequential inference on a list of audio filepath(s), processed one at a time, using a particular combination of ASR model and decoder."""

        transcripts = []
        for filepath in tqdm(filepaths, total=len(filepaths), unit=" transcript", desc="Generating transcripts predictions sequentially, so far"):
            emission_mx = self.model.forward([filepath], self.device)
            transcript = self.decoder.generate(emission_mx)[0]
            transcripts.append(transcript)
        
        # for transcript in transcripts:
        #     print(transcript)

        return transcripts

    def infer(self, filepaths: List[str], batch_size=0) -> List[str]:
        """Performs minibatched inference on a list of audio sample(s) specified by the filepath(s), using a particular combination of ASR model and decoder.
        If batch_size=0, performs sequential inference where audio samples are processed one at a time.
        If batch_size=len(filepaths), performs batched inference where all audio samples are processed at one time as a single input matrix.
        If 0 < batch_size < len(filepaths), performs minibatch inference.
        Only ASR inference is batched, while audio preprocessing and decoding is done sequentially, sample by sample."""
        # initialise the transcripts list for all files
        transcripts = []

        if batch_size == 0 or batch_size == 1 or self.model.vocab_path_or_bundle.startswith('torchaudio'):
            # sequential inference
            transcripts = self._infer_sequential(filepaths)
        else:
            # minibatch inference
            assert batch_size <= len(filepaths), "ERROR: batch_size must be less than or equal to the number of audio samples to process for inference."
            for i in tqdm(range(0, len(filepaths), batch_size), total=int(math.ceil(len(filepaths)/batch_size)), unit=" minibatch", desc="Generating transcripts in minibatches, so far"):
                emission_mx = self.model.forward(filepaths[i:i+batch_size], self.device)
                transcripts.append(self.decoder.generate(emission_mx))

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

    # returns a torchaudio wav2vec2 model and a greedy decoder.
    @classmethod
    def get_torchaudio_greedy(cls, bundle_str: str = 'torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H') -> ASR_Decoder_Pair:
        # using largest available wav2vec2 model from torchaudio by default
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(TorchaudioWav2Vec2Model(device=device, vocab_path_or_bundle=bundle_str),
                                GreedyDecoder(vocab_path_or_bundle=bundle_str))

    # returns a torchaudio wav2vec2 model and a beam search decoder coupled with a KenLM language model also from torchaudio.
    @classmethod
    def get_torchaudio_beamsearchkenlm(cls, bundle_str: str = 'torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H') -> ASR_Decoder_Pair:
        # using largest available wav2vec2 model from torchaudio
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(TorchaudioWav2Vec2Model(device=device, vocab_path_or_bundle=bundle_str),
                                BeamSearchKenLMDecoder_Torch(vocab_path_or_bundle=bundle_str))

    # returns a wav2vec2 model loaded from a custom checkpoint that has an args field and a Viterbi decoder from fairseq.
    @classmethod
    def get_args_viterbi(cls, model_filepath: str, vocab_path: str) -> ASR_Decoder_Pair:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(ArgsWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                ViterbiDecoder(vocab_path_or_bundle=vocab_path))

    # returns a wav2vec2 model loaded from a custom checkpoint that has a cfg field and a Viterbi decoder from fairseq.
    @classmethod
    def get_cfg_viterbi(cls, model_filepath: str, vocab_path: str) -> ASR_Decoder_Pair:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(CfgWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                ViterbiDecoder(vocab_path_or_bundle=vocab_path))

    # returns a wav2vec2 model loaded from a custom checkpoint that has an args field and a greedy decoder.
    @classmethod
    def get_args_greedy(cls, model_filepath: str, vocab_path: str) -> ASR_Decoder_Pair:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(ArgsWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                GreedyDecoder(vocab_path_or_bundle=vocab_path))

    # returns a wav2vec2 model loaded from a custom checkpoint that has a cfg field and a greedy decoder.
    @classmethod
    def get_cfg_greedy(cls, model_filepath: str, vocab_path: str) -> ASR_Decoder_Pair:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(CfgWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                GreedyDecoder(vocab_path_or_bundle=vocab_path))

    # returns a wav2vec2 model loaded from a custom checkpoint that has an args field and a beam search decoder with a KenLM language model from torchaudio.
    @classmethod
    def get_args_beamsearchkenlm_torch(cls, model_filepath: str, vocab_path: str) -> ASR_Decoder_Pair:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(ArgsWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                BeamSearchKenLMDecoder_Torch(vocab_path_or_bundle=vocab_path))

    # returns a wav2vec2 model loaded from a custom checkpoint that has a cfg field and a beam search decoder with a KenLM language model from torchaudio.
    @classmethod
    def get_cfg_beamsearchkenlm_torch(cls, model_filepath: str, vocab_path: str) -> ASR_Decoder_Pair:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(CfgWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                BeamSearchKenLMDecoder_Torch(vocab_path_or_bundle=vocab_path))

    # returns a wav2vec2 model loaded from a custom checkpoint that has an args field and a beam search decoder with a KenLM language model from fairseq.
    @classmethod
    def get_args_beamsearchkenlm_fairseq(cls, model_filepath: str, vocab_path: str) -> ASR_Decoder_Pair:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(ArgsWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                BeamSearchKenLMDecoder_Fairseq(vocab_path_or_bundle=vocab_path))

    # returns a wav2vec2 model loaded from a custom checkpoint that has a cfg field and a beam search decoder with a KenLM language model from fairseq.
    @classmethod
    def get_cfg_beamsearchkenlm_fairseq(cls, model_filepath: str, vocab_path: str) -> ASR_Decoder_Pair:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(CfgWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                BeamSearchKenLMDecoder_Fairseq(vocab_path_or_bundle=vocab_path))

    # returns a wav2vec2 model loaded from a custom checkpoint that has an args field and a beam search decoder with a Transformer language model from fairseq.
    @classmethod
    def get_args_beamsearchtransformerlm(cls, model_filepath: str, vocab_path: str) -> ASR_Decoder_Pair:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(ArgsWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                TransformerDecoder(vocab_path_or_bundle=vocab_path))

    # returns a wav2vec2 model loaded from a custom checkpoint that has a cfg field and a beam search decoder with a Transformer language model from fairseq.
    @classmethod
    def get_cfg_beamsearchtransformerlm(cls, model_filepath: str, vocab_path: str) -> ASR_Decoder_Pair:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(CfgWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                TransformerDecoder(vocab_path_or_bundle=vocab_path))


def main() -> None:
    # model configs
    bundle_str = "torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H"
    # wav2vec2 models checkpoints that were trained in the fairseq framework
    args_model_filepath = "/workspace/projects/Alignment/wav2vec2_alignment/Models/w2v_fairseq/wav2vec2_vox_960h_new.pt"
    cfg_model_filepath = "/workspace/projects/Alignment/wav2vec2_alignment/Models/vox_55h/checkpoints/checkpoint_best.pt"
    # vocab dicts used during the training of the corresponding wav2vec2 models trained in the fairseq framework
    args_vocab_filepath = "/workspace/projects/Alignment/wav2vec2_alignment/Models/w2v_fairseq/dict.ltr.txt"
    cfg_vocab_filepath = "/workspace/projects/Alignment/wav2vec2_alignment/Models/vox_55h/dict.ltr.txt"
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