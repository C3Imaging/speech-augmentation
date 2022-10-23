import torch
import soundfile as sf
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
    fields_dict = {}
    fields = Wav2Vec2CtcConfig.__dataclass_fields__
    # create dict for attributes of Wav2Vec2CtcConfig with vals taken from the same key in args
    for field in fields.keys():
        if field in args:
            fields_dict[field] = args[field]

    # fields_dict['w2v_path'] = model_path
    assert len(fields_dict) == len(fields)

    return fields_dict


def get_feature(filepath):
    def postprocess(feats, sample_rate):
        if feats.dim == 2:
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()

        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
        return feats

    wav, sample_rate = sf.read(filepath)
    feats = torch.from_numpy(wav).float()
    feats = postprocess(feats, sample_rate)
    feats = feats.cuda()

    return feats
