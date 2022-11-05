# using principles of dependency injection and dependency inversion and creating factories to simplify API of using wav2vec2 models loaded in different ways + various decoders for ASR inference

import torch
import re
import os
import torchaudio
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import decoding_utils_chkpt, decoding_utils_torch
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc, Wav2Vec2CtcConfig
from fairseq.data import Dictionary
from fairseq.data.data_utils import post_process
from argparse import Namespace
from omegaconf import OmegaConf
from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder


# wav2vec2 ASR model abstract class
class BaseWav2Vec2Model(ABC):
    # show what instance attributes should be defined
    model: torch.nn.Module
    vocab_path_or_bundle: str
    sample_rate: int

    @abstractmethod
    def __init__(self, device: torch.device, model_filepath: str, vocab_path_or_bundle: str) -> None:
        """Constructor for a model from a checkpoint (.pt) filepath or a torchaudio bundle."""
        self.vocab_path_or_bundle = vocab_path_or_bundle
    
    @abstractmethod
    def forward(self, audio_sample: torch.Tensor) -> Any:
        """Runs inference on a single preprocessed audio input, returning the prediction."""


# wav2vec2 ASR models (concrete implementations)
class TorchaudioWav2Vec2Model(BaseWav2Vec2Model):
    def __init__(self, device: torch.device, model_filepath: str = '', vocab_path_or_bundle: str = '') -> None:
        super().__init__(device, model_filepath, vocab_path_or_bundle)
        # if the passed in string is a torchaudio bundle
        assert re.match(r'torchaudio.pipelines', vocab_path_or_bundle) is not None, "ERROR!!! Please specify the torchaudio bundle to use as this wav2vec2 model."
        bundle = eval(vocab_path_or_bundle)
        self.model = bundle.get_model().to(device)
        self.sample_rate = bundle.sample_rate
        
    def forward(self, audio_sample: torch.Tensor) -> Any:
        """Inference is called directly in torchaudio to get the emissions matrix (the prediction) from the ASR model."""
        return self.model(audio_sample)


class ArgsWav2Vec2Model(BaseWav2Vec2Model):
    def __init__(self, device: torch.device, model_filepath: str, vocab_path_or_bundle: str) -> None:
        super().__init__(device, model_filepath, vocab_path_or_bundle)

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

    def forward(self, audio_sample: torch.Tensor) -> Any:
        """Inference is not called directly in the fairseq framework but is done through a decoder."""
        pass


class CfgWav2Vec2Model(BaseWav2Vec2Model):
    def __init__(self, device: torch.device, model_filepath: str, vocab_path_or_bundle: str) -> None:
        super().__init__(device, model_filepath, vocab_path_or_bundle)

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

    def forward(self, audio_sample: torch.Tensor) -> Any:
        """Inference is not called directly in the fairseq framework but is done through a decoder."""
        pass


# ASR decoder abstract class
class BaseDecoder(ABC):
    # show what instance attributes should be defined
    decoder: Any # different decoders do not have a common interface

    @abstractmethod
    def _preprocess_audiofile(self, filepath: str, device: torch.device, trg_sample_rate: int) -> torch.Tensor:
        """Since in the fairseq framework inference is not called on an ASR model directly, but is done through a decoder, the decoder will be responsible for preprocessing the
            audio sample in the required format for the particular ASR acoustic model being used. This keeps the interface consistent, even for torchaudio ASR acoustic models."""

    @abstractmethod
    def generate(self, model: BaseWav2Vec2Model, filepath: str, device: torch.device) -> str:
        """Generates a transcript. Involves calling inference on the ASR acoustic model and decoding the emission matrix outputted by it."""

    @staticmethod
    def _get_vocab(vocab_path_or_bundle: str):
        # if the passed in string is a torchaudio bundle
        if re.match(r'torchaudio.pipelines', vocab_path_or_bundle):
            bundle = eval(vocab_path_or_bundle)
            vocab =  [label.lower() for label in bundle.get_labels()]
            return vocab
        # else the passed in string is a text file with a vocabulary
        target_dict = Dictionary.load(vocab_path_or_bundle)
        return target_dict.symbols


# ASR decoders (concrete implementations)
class GreedyDecoder(BaseDecoder):
    # tested to work with a torchaudio wav2vec2 ASR model
    def __init__(self, vocab_path_or_bundle: str) -> None:
        # the vocabulary of chars known by the acoustic model that will be used for decoding
        vocab = self._get_vocab(vocab_path_or_bundle)
        self.decoder = decoding_utils_torch.GreedyCTCDecoder(vocab)

    def _preprocess_audiofile(self, filepath: str, device: torch.device, trg_sample_rate: int) -> torch.Tensor:
        # load audio file and resample if needed
        waveform, sr = torchaudio.load(filepath)
        if sr != trg_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, trg_sample_rate)
        return waveform.to(device)

    def generate(self, model: BaseWav2Vec2Model, filepath: str, device: torch.device) -> str:
        # generate one transcript
        waveform = self._preprocess_audiofile(filepath=filepath, device=device, trg_sample_rate=model.sample_rate,)
        emissions, _ = model.model.forward(waveform)
        result = self.decoder(emissions) # decoded phrase as a list of words
        transcript = ' '.join(result)

        return transcript


class BeamSearchKenLMDecoder(BaseDecoder):
    # tested to work with a torchaudio TorchaudioWav2Vec2Model ASR model
    def __init__(self, vocab_path_or_bundle: str) -> None:
        vocab_from_txt = False # just a flag, True if using a wav2vec2 model from a checkpoint file
        vocab = self._get_vocab(vocab_path_or_bundle)
        # if the vocab came from a txt file, to use KenLM, we need to remove the <s>, <pad>, </s>, <unk> chars added by fairseq from the vocab list
        if vocab[0] == '<s>':
            vocab = vocab[4:]
            vocab_from_txt = True
            # now need to save the vocab list to a txt file with newlines between chars
            with open(os.path.join(os.getcwd(), 'tokens.txt'), 'w') as f:
                for i in range(len(vocab)):
                    f.write(vocab[i].lower())
                    # do not add a newline after the last char
                    if i != len(vocab)-1:
                        f.write('\n')

        # get KenLM language model config
        files = torchaudio.models.decoder.download_pretrained_files("librispeech-4-gram")

        tokens_path = os.path.join(os.getcwd(), 'tokens.txt') if vocab_from_txt else files.tokens

        # in this case, the vocab is taken from the KenLM language model
        # TODO: I think files.tokens should actually be replaced with target_dict.symbols if not using torchaudio wav2vec2
        # initialise a beam search decoder with a KenLM language model
        self.decoder = torchaudio.models.decoder.ctc_decoder(
            lexicon=files.lexicon, # giant file of English "words"
            tokens=tokens_path, # same as wav2vec2's vocab
            lm=files.lm, # path to language model binary
            nbest=3,
            beam_size=1500,
            lm_weight=3.23,
            word_score=-0.26,
        )
    
    def _preprocess_audiofile(self, filepath: str, device: torch.device, trg_sample_rate: int) -> torch.Tensor:
        # load audio file and resample if needed
        waveform, sr = torchaudio.load(filepath)
        if sr != trg_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, trg_sample_rate)
        return waveform.to(device)

    def generate(self, model: BaseWav2Vec2Model, filepath: str, device: torch.device) -> str:
        waveform = self._preprocess_audiofile(filepath=filepath, device=device, trg_sample_rate=model.sample_rate)
        emissions, _ = model.model.forward(waveform)
        result = self.decoder(emissions.cpu().detach())[0][0].words # decoded phrase as a list of words
        transcript = ' '.join(result)

        return transcript


class ViterbiDecoder(BaseDecoder):
    # tested to work under the fairseq framework
    def __init__(self, vocab_path_or_bundle: str) -> None:
        assert re.match(r'torchaudio.pipelines', vocab_path_or_bundle) is None, "Cannot provide a torch bundle to ViterbiDecoder, must use a txt file."
        target_dict = Dictionary.load(vocab_path_or_bundle)
        # define additional decoder args
        decoder_args = Namespace(**{'nbest': 1})
        self.decoder = W2lViterbiDecoder(decoder_args, target_dict)

    def _preprocess_audiofile(self, filepath: str, device: torch.device, trg_sample_rate: int) -> torch.Tensor:
        # load audio file and resample if needed
        waveform, sr = decoding_utils_chkpt.get_feature(filepath)
        if sr != trg_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, trg_sample_rate)

        return waveform.to(device)

    def generate(self, model: BaseWav2Vec2Model, filepath: str, device: torch.device) -> str:
        waveform = self._preprocess_audiofile(filepath=filepath, device=device, trg_sample_rate=model.sample_rate)
        sample, input = dict(), dict()
        input["source"] = waveform.unsqueeze(0)
        padding_mask = torch.BoolTensor(input["source"].size(1)).fill_(False).unsqueeze(0)
        input["padding_mask"] = padding_mask
        sample["net_input"] = input

        models = list()
        models.append(model.model)

        with torch.no_grad():
            hypo = self.decoder.generate(models, sample, prefix_tokens=None)

        target_dict = Dictionary.load(model.vocab_path_or_bundle)
        hyp_pieces = target_dict.string(hypo[0][0]["tokens"].int().cpu())
        transcript = post_process(hyp_pieces, 'letter')

        return transcript


# ASR factory interface - used to return a particular combination of an ASR acoustic model and an ASR decoder
class Wav2Vec2_Decoder_Factory(ABC):
    # purely an interface - no need to create instances of factories as there are no instance attributes that need to be defined.
    @classmethod
    @abstractmethod
    def get_asr_model_and_decoder(cls) -> Tuple[BaseWav2Vec2Model, BaseDecoder]:
        """Returns a tuple: (BaseWav2Vec2Model, BaseDecoder)
            - wav2vec2 ASR model object is of type BaseWav2Vec2Model to keep the interface consistent, regardless of how the model was loaded.
            - ASR decoder object is of type BaseDecoder to keep the interface consistent, regardless of what decoder is used."""


# ASR factories (concrete implementations)
class TorchaudioWav2Vec2_GreedyDecoder_Factory(Wav2Vec2_Decoder_Factory):
    # returns a torchaudio wav2vec2 model and a greedy decoder.
    # specify manually what bundle to use for the wav2vec2 model.
    @classmethod
    def get_asr_model_and_decoder(cls) -> Tuple[BaseWav2Vec2Model, BaseDecoder]:
        bundle_str = 'torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H' # using largest available wav2vec2 model from torchaudio

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return TorchaudioWav2Vec2Model(device=device, vocab_path_or_bundle=bundle_str), GreedyDecoder(vocab_path_or_bundle=bundle_str)


class TorchaudioWav2Vec2_BeamSearchKenLMDecoder_Factory(Wav2Vec2_Decoder_Factory):
    # returns a torchaudio wav2vec2 model and a beam search decoder coupled with a KenLM language model.
    @classmethod
    def get_asr_model_and_decoder(cls) -> Tuple[BaseWav2Vec2Model, BaseDecoder]:
        # specify manually what bundle to use for the wav2vec2 model.
        bundle_str = 'torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H' # using largest available wav2vec2 model from torchaudio

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return TorchaudioWav2Vec2Model(device=device, vocab_path_or_bundle=bundle_str), BeamSearchKenLMDecoder(vocab_path_or_bundle=bundle_str)


class ArgsWav2Vec2_ViterbiDecoder_Factory(Wav2Vec2_Decoder_Factory):
    # returns a wav2vec2 model loaded from a custom checkpoint that has an args field and a Viterbi decoder.
    @classmethod
    def get_asr_model_and_decoder(cls) -> Tuple[BaseWav2Vec2Model, BaseDecoder]:
        # manually specify the path to the checkpoint
        model_filepath = "/workspace/projects/Alignment/wav2vec2_alignment/Models/w2v_fairseq/wav2vec2_vox_960h_new.pt"
        # manually specify the path to the vocab of the model
        vocab_path = "/workspace/projects/Alignment/wav2vec2_alignment/Models/w2v_fairseq/dict.ltr.txt"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return ArgsWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path), ViterbiDecoder(vocab_path_or_bundle=vocab_path)


class CfgWav2Vec2_ViterbiDecoder_Factory(Wav2Vec2_Decoder_Factory):
    # returns a wav2vec2 model loaded from a custom checkpoint that has a cfg field and a Viterbi decoder.
    @classmethod
    def get_asr_model_and_decoder(cls) -> Tuple[BaseWav2Vec2Model, BaseDecoder]:
        # manually specify the path to the checkpoint
        model_filepath = "/workspace/projects/Alignment/wav2vec2_alignment/Models/vox_55h/checkpoints/checkpoint_best.pt"
        # manually specify the path to the vocab of the model
        vocab_path = "/workspace/projects/Alignment/wav2vec2_alignment/Models/vox_55h/dict.ltr.txt"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return CfgWav2Vec2Model(device=device, model_filepath=model_filepath, vocab_path_or_bundle=vocab_path), ViterbiDecoder(vocab_path_or_bundle=vocab_path)


def test_main():
    # TODO: make proper testing functions
    # test inference on one audio file
    wav_filepath = "/workspace/datasets/myst_test/myst_999465_2009-17-12_00-00-00_MS_4.2_024.wav"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test torchaudio wav2vec2 + greedy decoder -> works
    # acoustic_model, decoder = TorchaudioWav2Vec2_GreedyDecoder_Factory.get_asr_model_and_decoder()
    # transcript = decoder.generate(acoustic_model, wav_filepath, device)
    # print(transcript)

    # test torchaudio wav2vec2 + beam search decoder with KenLM language model -> works
    # acoustic_model, decoder = TorchaudioWav2Vec2_BeamSearchKenLMDecoder_Factory.get_asr_model_and_decoder()
    # transcript = decoder.generate(acoustic_model, wav_filepath, device)
    # print(transcript)

    # test args wav2vec2 + viterbi -> works
    # acoustic_model, decoder = CfgWav2Vec2_ViterbiDecoder_Factory.get_asr_model_and_decoder()
    # transcript = decoder.generate(acoustic_model, wav_filepath, device)
    # print(transcript)

    # test cfg wav2vec2 + viterbi -> works
    # acoustic_model, decoder = CfgWav2Vec2_ViterbiDecoder_Factory.get_asr_model_and_decoder()
    # transcript = decoder.generate(acoustic_model, wav_filepath, device)
    # print(transcript)


if __name__ == "__main__":
    test_main()