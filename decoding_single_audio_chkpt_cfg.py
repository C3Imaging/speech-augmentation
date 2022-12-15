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
from Tools.decoding_utils_chkpt import get_features_list, get_padded_batch_mxs, get_config_dict


if __name__ == "__main__":
    model_path = "/workspace/projects/Alignment/wav2vec2_alignment/Models/vox_55h/checkpoints/checkpoint_best.pt"
    target_dict = Dictionary.load('/workspace/projects/Alignment/wav2vec2_alignment/Models/vox_55h/dict.ltr.txt')

    w2v = torch.load(model_path)

    args_dict = get_config_dict(w2v['cfg']['model'])
    w2v_config_obj = OmegaConf.merge(OmegaConf.structured(Wav2Vec2CtcConfig), args_dict)

    dummy_target_dict = {'target_dictionary' : target_dict.symbols}
    dummy_target_dict = Namespace(**dummy_target_dict)

    model = Wav2VecCtc.build_model(w2v_config_obj, dummy_target_dict)
    model.load_state_dict(w2v["model"], strict=True)
    model = model.cuda()
    model.eval()

    sample, input = dict(), dict()
    WAV_PATH1 = '/workspace/datasets/myst_test/myst_999465_2009-17-12_00-00-00_MS_4.2_024.wav'
    WAV_PATH2 = '/workspace/datasets/myst_test/myst_002030_2014-02-28_09-37-51_LS_1.1_006.wav'
    # NOTE: for Viterbi decoding, the frequencies of vocab labels from dataset used for finetuning wav2vec2 model is not needed
    # define additional decoder args
    decoder_args = Namespace(**{'nbest': 1})
    generator = W2lViterbiDecoder(decoder_args, target_dict)

    features, sr = get_features_list([WAV_PATH1, WAV_PATH2])
    padded_features, padding_masks = get_padded_batch_mxs(features)
    padded_features = padded_features.cuda()

    #padding_mask = torch.BoolTensor(input["source"].size(1)).fill_(False).unsqueeze(0)

    input["source"] = padded_features
    input["padding_mask"] = padding_masks
    sample["net_input"] = input

    models = list()
    models.append(model)

    with torch.no_grad():
        hypos = generator.generate(models, sample, prefix_tokens=None)

    transcripts = [post_process(target_dict.string(h[0]['tokens'].int().cpu()), 'letter') for h in hypos]

    print(transcripts)