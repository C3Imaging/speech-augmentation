# using principles of dependency injection and dependency inversion and creating factories to simplify API of using wav2vec2 models loaded in different ways + various decoders for ASR inference

import re
import torch
import torchaudio
from typing import List, Any
from argparse import Namespace
from omegaconf import OmegaConf
from abc import ABC, abstractmethod
from fairseq.data import Dictionary
import decoding_utils_chkpt, decoding_utils_torch
from fairseq.data.data_utils import post_process
from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc, Wav2Vec2CtcConfig


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
    def forward(self, filepath: str, device: torch.device) -> torch.Tensor:
        """Runs inference on a single audio file input, first preprocessing the audio as required by the model and then returning the emissions matrix."""


# wav2vec2 ASR models (concrete implementations)
class TorchaudioWav2Vec2Model(BaseWav2Vec2Model):
    def __init__(self, device: torch.device, vocab_path_or_bundle: str = '') -> None:
        super().__init__(vocab_path_or_bundle)
        # if the passed in string is a torchaudio bundle
        assert re.match(r'torchaudio.pipelines', vocab_path_or_bundle) is not None, "ERROR!!! Please specify the torchaudio bundle to use as this wav2vec2 model."
        bundle = eval(vocab_path_or_bundle)
        self.model = bundle.get_model().to(device)
        self.sample_rate = bundle.sample_rate

    def forward(self, filepath: str, device: torch.device) -> torch.Tensor:
        waveforms, sr = torchaudio.load(filepath)
        if sr != self.sample_rate:
            # torchaudio loads even a single sample as a batch with B=1
            waveforms = torchaudio.functional.resample(waveforms, sr, self.sample_rate)
        waveform = waveforms.to(device)[0]

        emissions, _ = self.model(waveform.unsqueeze(0)) # add a batch dimension
        emission_mx = emissions[0]

        return emission_mx


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

    def forward(self, filepath: str, device: torch.device) -> torch.Tensor:
        """Inference is not called directly in the fairseq framework API but is done through a decoder, but I copy-pasted the actual inference part from
            fairseq/examples/speech_recognition/w2l_decoder.W2lDecoder.generate()"""
        waveform, sr = decoding_utils_chkpt.get_feature(filepath)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.to(device)

        sample, input = dict(), dict()
        input["source"] = waveform.unsqueeze(0)
        padding_mask = torch.BoolTensor(input["source"].size(1)).fill_(False).unsqueeze(0)
        input["padding_mask"] = padding_mask
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
        emission_mx = emissions[0]

        return emission_mx


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

    def forward(self, filepath: str, device: torch.device) -> torch.Tensor:
        """Inference is not called directly in the fairseq framework API but is done through a decoder, but I copy-pasted the actual inference part from
            fairseq/examples/speech_recognition/w2l_decoder.W2lDecoder.generate()"""
        waveform, sr = decoding_utils_chkpt.get_feature(filepath)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.to(device)

        sample, input = dict(), dict()
        input["source"] = waveform.unsqueeze(0)
        padding_mask = torch.BoolTensor(input["source"].size(1)).fill_(False).unsqueeze(0)
        input["padding_mask"] = padding_mask
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
        emission_mx = emissions[0]

        return emission_mx


# ASR decoder abstract class
class BaseDecoder(ABC):
    # show what instance attributes should be defined
    decoder: Any # different decoders do not have a common interface

    @abstractmethod
    def generate(self, emission_mx: torch.Tensor) -> str:
        """Generates a transcript by decoding the output of a wav2vec2 ASR acoutic model."""

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

    def generate(self, emission_mx: torch.Tensor) -> str:
        # generate one transcript
        # decoded phrase as a list of words
        result = self.decoder(emission_mx.unsqueeze(0)) # add a batch dimension
        transcript = ' '.join(result)

        return transcript.lower()


class BeamSearchKenLMDecoder(BaseDecoder):
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

    def generate(self, emission_mx: torch.Tensor) -> str:
        emissions = emission_mx.unsqueeze(0) # add a batch dimension
        # select the decoded phrase as a list of words from the prediction of the first beam (most likely transcript)
        result = self.decoder(emissions.cpu().detach())[0][0].words 
        transcript = ' '.join(result)

        return transcript.lower()


class ViterbiDecoder(BaseDecoder):
    """The Viterbi algorithm is not a greedy algorithm.
    It performs a global optimisation and guarantees to find the most likely state sequence, by exploring all possible state sequences.
    """
    def __init__(self, vocab_path_or_bundle: str) -> None:
        assert re.match(r'torchaudio.pipelines', vocab_path_or_bundle) is None, "Cannot provide a torch bundle to ViterbiDecoder, must use a txt file."
        target_dict = Dictionary.load(vocab_path_or_bundle)
        # define additional decoder args
        decoder_args = Namespace(**{'nbest': 1})
        # vocab is passed to the decoder object during initialisation
        self.decoder = W2lViterbiDecoder(decoder_args, target_dict)

    def generate(self, emission_mx: torch.Tensor) -> str:
        hypo = self.decoder.decode(emission_mx.unsqueeze(0)) # add a batch dimension
        hyp_pieces = self.decoder.tgt_dict.string(hypo[0][0]["tokens"].int().cpu())
        transcript = post_process(hyp_pieces, 'letter')

        return transcript.lower()


class ASR_Decoder_Pair():
    """A bundle representing a particular combination of an ASR model and decoder.
    The combination is stored in this single object to prevent using an ASR model and decoder that are incompatible or that were initialised with different vocabs."""

    def __init__(self, model: BaseWav2Vec2Model, decoder: BaseDecoder) -> None:
        self.model = model
        self.decoder = decoder

    def infer(self, filepaths: List[str]) -> List[str]:
        """performs single or batched inference on a list of audio filepath(s) using a particular combination of ASR model and decoder"""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transcripts = []
        for filepath in filepaths:
            emission_mx = self.model.forward(filepath, device)
            transcript = self.decoder.generate(emission_mx)
            transcripts.append(transcript)
        
        for transcript in transcripts:
            print(transcript)

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

    # returns a torchaudio wav2vec2 model and a beam search decoder coupled with a KenLM language model.
    @classmethod
    def get_torchaudio_beamsearchkenlm(cls, bundle_str: str = 'torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H') -> ASR_Decoder_Pair:
        # using largest available wav2vec2 model from torchaudio
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(TorchaudioWav2Vec2Model(device=device, vocab_path_or_bundle=bundle_str),
                                BeamSearchKenLMDecoder(vocab_path_or_bundle=bundle_str))

    # returns a wav2vec2 model loaded from a custom checkpoint that has an args field and a Viterbi decoder.
    @classmethod
    def get_args_viterbi(cls, model_filepath: str, vocab_path: str) -> ASR_Decoder_Pair:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(ArgsWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                ViterbiDecoder(vocab_path_or_bundle=vocab_path))

    # returns a wav2vec2 model loaded from a custom checkpoint that has a cfg field and a Viterbi decoder.
    @classmethod
    def get_cfg_viterbi(cls, model_filepath: str, vocab_path: str) -> ASR_Decoder_Pair:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(CfgWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                ViterbiDecoder(vocab_path_or_bundle=vocab_path))

    # returns a wav2vec2 model loaded from a custom checkpoint that has an args field and a beam search decoder with a KenLM language model.
    @classmethod
    def get_args_greedy(cls, model_filepath: str, vocab_path: str) -> ASR_Decoder_Pair:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(ArgsWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                GreedyDecoder(vocab_path_or_bundle=vocab_path))

    # returns a wav2vec2 model loaded from a custom checkpoint that has a cfg field and a beam search decoder with a KenLM language model.
    @classmethod
    def get_cfg_greedy(cls, model_filepath: str, vocab_path: str) -> ASR_Decoder_Pair:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(CfgWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                GreedyDecoder(vocab_path_or_bundle=vocab_path))

    # returns a wav2vec2 model loaded from a custom checkpoint that has an args field and a beam search decoder with a KenLM language model.
    @classmethod
    def get_args_beamsearchkenlm(cls, model_filepath: str, vocab_path: str) -> ASR_Decoder_Pair:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(ArgsWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                BeamSearchKenLMDecoder(vocab_path_or_bundle=vocab_path))

    # returns a wav2vec2 model loaded from a custom checkpoint that has a cfg field and a beam search decoder with a KenLM language model.
    @classmethod
    def get_cfg_beamsearchkenlm(cls, model_filepath: str, vocab_path: str) -> ASR_Decoder_Pair:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASR_Decoder_Pair(CfgWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path),
                                BeamSearchKenLMDecoder(vocab_path_or_bundle=vocab_path))


def main() -> None:
    # model configs
    bundle_str = "torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H"
    args_model_filepath = "/workspace/projects/Alignment/wav2vec2_alignment/Models/w2v_fairseq/wav2vec2_vox_960h_new.pt"
    cfg_model_filepath = "/workspace/projects/Alignment/wav2vec2_alignment/Models/vox_55h/checkpoints/checkpoint_best.pt"
    args_vocab_filepath = "/workspace/projects/Alignment/wav2vec2_alignment/Models/w2v_fairseq/dict.ltr.txt"
    cfg_vocab_filepath = "/workspace/projects/Alignment/wav2vec2_alignment/Models/vox_55h/dict.ltr.txt"
    # audio samples to test inference on
    wavpaths = ["/workspace/datasets/myst_test/myst_999465_2009-17-12_00-00-00_MS_4.2_024.wav",
                "/workspace/datasets/myst_test/myst_002030_2014-02-28_09-37-51_LS_1.1_006.wav"]

    # test torchaudio wav2vec2 + greedy decoder -> works
    asr1 = Wav2Vec2_Decoder_Factory.get_torchaudio_greedy(bundle_str=bundle_str)
    transcripts1 = asr1.infer(wavpaths)

    # test torchaudio wav2vec2 + beam search decoder with KenLM language model -> works
    asr2 = Wav2Vec2_Decoder_Factory.get_torchaudio_beamsearchkenlm(bundle_str=bundle_str)
    transcripts2 = asr2.infer(wavpaths)

    # test args wav2vec2 + viterbi -> works
    asr3 = Wav2Vec2_Decoder_Factory.get_args_viterbi(model_filepath=args_model_filepath, vocab_path=args_vocab_filepath)
    transcripts3 = asr3.infer(wavpaths)

    # test cfg wav2vec2 + viterbi -> works
    asr4 = Wav2Vec2_Decoder_Factory.get_cfg_viterbi(model_filepath=cfg_model_filepath, vocab_path=cfg_vocab_filepath)
    transcripts4 = asr4.infer(wavpaths)

    # test args wav2vec2 + greedy decoder -> works
    asr5 = Wav2Vec2_Decoder_Factory.get_args_greedy(model_filepath=args_model_filepath, vocab_path=args_vocab_filepath)
    transcripts5 = asr5.infer(wavpaths)

    # test cfg wav2vec2 + greedy decoder -> works
    asr6 = Wav2Vec2_Decoder_Factory.get_cfg_greedy(model_filepath=cfg_model_filepath, vocab_path=cfg_vocab_filepath)
    transcripts6 = asr6.infer(wavpaths)

    # test args wav2vec2 + beam search decoder with KenLM language model -> works
    asr7 = Wav2Vec2_Decoder_Factory.get_args_beamsearchkenlm(model_filepath=args_model_filepath, vocab_path=args_vocab_filepath)
    transcripts7 = asr7.infer(wavpaths)

    # test cfg wav2vec2 + beam search decoder with KenLM language model -> works
    asr8 = Wav2Vec2_Decoder_Factory.get_cfg_beamsearchkenlm(model_filepath=cfg_model_filepath, vocab_path=cfg_vocab_filepath)
    transcripts8 = asr8.infer(wavpaths)


if __name__ == "__main__":
    main()