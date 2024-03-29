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


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emissions: torch.Tensor, labels=None, blank=0):
        """Given a sequence emission over labels, get the best path
        Args:
        emissions (List[tensor]): Logit tensors. Shape `[batch_size, num_seq, num_label]`.
        labels (List[str]): Vocabulary of the wav2vec2 model that produced the emissions tensor.
        blank (int): blank token from CTC formulation.

        Returns:
        List[str]: The resulting transcript
        """
        emission = emissions[0]
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != blank]
        joined = "".join([self.labels[i] for i in indices])

        return joined.replace("|", " ").strip().split()
