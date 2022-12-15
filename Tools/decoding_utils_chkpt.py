import torch
import torchaudio
from argparse import Namespace
import torch.nn.functional as F
from fairseq.models.wav2vec.wav2vec2_asr import Wav2Vec2CtcConfig

# from flashlight.lib.text.decoder import CriterionType
# from flashlight.lib.sequence.criterion import CpuViterbiPath
# from flashlight.lib.sequence.criterion_torch import get_data_ptr_as_bytes


def get_config_dict(args):
    if isinstance(args, Namespace):
        # unpack Namespace into base dict obj
        args = vars(args)
    fields = Wav2Vec2CtcConfig.__dataclass_fields__
    # create dict for attributes of Wav2Vec2CtcConfig with vals taken from the same key in args, if they exist
    fields_dict = {}
    # this means Wav2Vec2CtcConfig obj fields will be overwritten with vals from args, otherwise they will be default
    for field in fields.keys():
        if field in args:
            fields_dict[field] = args[field]

    return fields_dict


def get_feature(filepath):
    # preprocess a single audio file for inference
    def postprocess(feats):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()

        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)

        return feats


    wavs, sample_rate = torchaudio.load(filepath)
    feats = wavs[0]
    feats = postprocess(feats)

    return feats, sample_rate


def get_features_list(filepaths):
    """Load and preprocess audio files into a list of 1D audio tensors."""
    def postprocess(feats):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()

        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)

        return feats

    inputs = []
    for i in range(len(filepaths)):
        wav, sample_rate = torchaudio.load(filepaths[i])
        feats = postprocess(wav[0])
        inputs.append(feats)


    return inputs, sample_rate


def get_padded_batch_mxs(tensor_list):
    """Convert list of 1D tensors of different lengths into a batch tensor where each tensor is zero padded to the length of the longest tensor.
    Args:
      tensor_list (List[str]):
        List of preprocessed 1D audio tensors.

    Returns:
      padded_batch_mx (torch.tensor):
        A batched tensor of padded audio tensors.
      padding_masks_mx (str, list):
        A batched tensor of padding tensors (masks) where values are False for corresponding audio values in the audio tensors and True for padding locations.
    """
    # get max length from the tensors list, pad all tensors to this length
    max_len = max(list(map(lambda t: t.size(dim=0), tensor_list)))
    padded_batch_mx = torch.zeros((len(tensor_list), max_len))
    padding_masks_mx = torch.ones((len(tensor_list), max_len), dtype=torch.bool)
    for i, t in enumerate(tensor_list):
        padded_batch_mx[i, :len(tensor_list[i])] = t
        padding_masks_mx[i, :len(tensor_list[i])] = False
    
    return padded_batch_mx, padding_masks_mx
