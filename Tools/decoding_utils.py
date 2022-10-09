import os
import torch
import torchaudio
import torchaudio.models.decoder
from abc import ABC, abstractmethod


# manually set the decoder classes visible when importing this module
known_decoders = ['GreedyCTCDecoder', 'BeamSearchDecoder']


def get_speech_data_lists(dirpath, filenames):
    """Gets the speech audio files paths and the txt files (if they exist) from a single folder.

    Args:
      dirpath (str):
        The path to the directory containing the audio files and some txt files, which the transcript files will be part of (if they exist).
      filenames (list of str elements):
        the list of all files found by os.walk in this directory.

    Returns:
      speech_files (str, list):
        A sorted list of speech file paths found in this directory.
      txt_files (str, list):
        A list of txt files.
    """
    speech_files = []
    txt_files = []
    # loop through all files found
    # split filenames in a directory into speech files and transcript files
    for filename in filenames:
        if filename.endswith('.wav'):
            speech_files.append(os.path.join(dirpath, filename))
        elif filename.endswith('.txt'):
            txt_files.append(os.path.join(dirpath, filename))
    # avoid sorting empty lists
    if len(speech_files):
        speech_files.sort()
    if len(txt_files):
        txt_files.sort()

    return speech_files, txt_files


# interface that mimics a torch.nn.Module by requiring a forward method implementation
class Decoder(ABC):
    @abstractmethod
    def forward(self, emissions: torch.Tensor):
        """Given a sequence emission over labels, get the best path
        Args:
            emissions (list, Tensor): Logit tensors. Shape `[batch_size, num_seq, num_label]`.

        Returns:
            List[str]: The resulting transcript
        """

class GreedyCTCDecoder(torch.nn.Module, Decoder):
    def __init__(self, labels=None, blank=0):
        super().__init__()
        if not labels:
            # hardcoded vocabulary if not provided externally
            self.labels = ('-','|','E','T','A','O','N','I','H','S','R','D','L','U','M','W','C','F','G','Y','P','B','V','K',"'",'X','J','Q','Z')
        else:
            self.labels = labels
        self.blank = blank

    def forward(self, emissions: torch.Tensor):
        emission = emissions[0]
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])

        return joined.replace("|", " ").strip().split()


class BeamSearchDecoder(Decoder):
    def __init__(self, lm_files=torchaudio.models.decoder.download_pretrained_files("librispeech-4-gram")):
        super().__init__()
        self.beam_search_decoder = torchaudio.models.decoder.ctc_decoder(
            lexicon=lm_files.lexicon, # giant file of English "words"
            tokens=lm_files.tokens, # same as wav2vec2's vocab list
            lm=lm_files.lm, # path to language model binary
            nbest=3,
            beam_size=1500,
            lm_weight=3.23,
            word_score=-0.26,
        )

    def forward(self, emissions: torch.Tensor):
        emission = emissions.cpu().detach()
        
        return self.beam_search_decoder(emission)[0][0].words