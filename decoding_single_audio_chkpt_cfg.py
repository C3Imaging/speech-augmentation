# run ASR inference using a wav2vec2 ASR model and a specified decoder on a single audio file.
# NOTE: this script can only use wav2vec2 ASR models from .pt checkpoint files.
# used for wav2vec2 ASR checkpoints that were finetuned in the Hydra framework (loaded checkpoint has 'cfg' key but no 'args' key).

import torch
from argparse import Namespace
from omegaconf import OmegaConf
from fairseq.data import Dictionary
from fairseq.data.data_utils import post_process
from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc, Wav2Vec2CtcConfig
from Tools.decoding_utils_chkpt import get_feature
from Tools.decoding_utils_chkpt import get_config_dict


if __name__ == "__main__":
    model_path = "/workspace/projects/Alignment/wav2vec2_alignment/Models/vox_55h/checkpoints/checkpoint_best.pt"
    target_dict = Dictionary.load('/workspace/projects/Alignment/wav2vec2_alignment/Models/vox_55h/dict.ltr.txt')

    w2v = torch.load(model_path)
    # override the path to the model checkpoint, not actually required
    # w2v['cfg']['model']['w2v_path'] = model_path
    # override pretraining task with finetuning task, not actually required
    #w2v['cfg']['model']['w2v_args']['task'] = w2v['cfg']['task']

    args_dict = get_config_dict(w2v['cfg']['model'])
    w2v_config_obj = OmegaConf.merge(OmegaConf.structured(Wav2Vec2CtcConfig), args_dict)

    dummy_target_dict = {'target_dictionary' : target_dict.symbols}
    dummy_target_dict = Namespace(**dummy_target_dict)

    model = Wav2VecCtc.build_model(w2v_config_obj, dummy_target_dict)
    model.load_state_dict(w2v["model"], strict=True)
    model = model.cuda()
    model.eval()

    sample, input = dict(), dict()
    WAV_PATH = '/workspace/datasets/myst_test/myst_999465_2009-17-12_00-00-00_MS_4.2_024.wav'

    # NOTE: for decoding, the frequencies of vocab labels from dataset used for finetuning is not needed
    # define additional decoder args
    decoder_args = Namespace(**{'nbest': 1})
    generator = W2lViterbiDecoder(decoder_args, target_dict)

    feature, _ = get_feature(WAV_PATH)
    feature = feature.cuda()
    input["source"] = feature.unsqueeze(0)

    padding_mask = torch.BoolTensor(input["source"].size(1)).fill_(False).unsqueeze(0)

    input["padding_mask"] = padding_mask
    sample["net_input"] = input

    models = list()
    models.append(model)

    with torch.no_grad():
        hypo = generator.generate(models, sample, prefix_tokens=None)

    hyp_pieces = target_dict.string(hypo[0][0]["tokens"].int().cpu())


    res = post_process(hyp_pieces, 'letter')
    print(res)