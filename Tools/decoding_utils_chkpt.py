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
    def postprocess(feats, sample_rate):
        if feats.dim == 2:
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()

        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
        return feats


    wavs, sample_rate = torchaudio.load(filepath)
    feats = wavs[0]
    feats = postprocess(feats, sample_rate)

    return feats, sample_rate
