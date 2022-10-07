import torch
import torchaudio
from typing import List
from abc import ABC, abstractmethod


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