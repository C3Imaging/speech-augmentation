import os
import sys
import json
import argparse

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Utils import utils
from typing import List, Any, Union, Dict
from Utils.asr.decoding_utils_w2v2 import Wav2Vec2_Decoder_Factory, ViterbiDecoder, GreedyDecoder, BaseDecoder
from Utils.asr.common_utils import Hypotheses, Hypothesis, WordAlign, write_results, get_all_wavs


def format_w2v2_output(w2v2_output: Union[List[List[Dict]], List[Dict]], time_aligns: bool, num_hyps: int, decoder: BaseDecoder) -> List[Hypotheses]:
    """
    Convert the output of a w2v2 decoder from 'Tools/asr/decoding_utils_w2v2.py' to a common format using dataclasses from 'Tools/asr/common_utils.py'.

    Args:
        w2v2_output (Union[List[List[Dict]], List[Dict]]):
            The output of decoding the output of Wav2Vec2 inference on a batch of audio files, with word-level timestamps if the decoder supports it.
            The format is common across different decoders available in 'Tools/asr/decoding_utils_w2v2.py', but uses built-in data structures.
        time_aligns (bool):
            Whether to include word-level time alignments in the results.
        num_hyps (int):
            The number of top hypotheses to keep in the results.
        decoder (BaseDecoder):
            The decoder used to produce the results. It will be an implementation class of the BaseDecoder abstract base class.
        
    Returns:
        results (List[Hypotheses]):
            The common format for results for any ASR inference output.
    """
    # loop through the result for each audio file.
    results = list()
    for audio_result in w2v2_output:
        hypotheses = list()
        if isinstance(audio_result, dict):
            # num_hyps = 1
            word_aligns = list()
            # GreedyDecoder ignores args.time_aligns
            if time_aligns and not isinstance(decoder, GreedyDecoder):
                for word in audio_result['timestamps_word']:
                    word_aligns.append(WordAlign(word['word'], word['start_time'], word['end_time']))
            hypotheses.append(Hypothesis(audio_result['pred_txt'], word_aligns))
        # GreedyDecoder and ViterbiDecoder ignore args.num_hyps
        elif not isinstance(decoder, (ViterbiDecoder, GreedyDecoder)):
            # num_hyps > 1
            for i in range(num_hyps):
                # create only as many Hypothesis objects as num_hyps.
                hypo = audio_result[i]
                word_aligns = list()
                if time_aligns:
                    for word in hypo['timestamps_word']:
                        word_aligns.append(WordAlign(word['word'], word['start_time'], word['end_time']))
                hypotheses.append(Hypothesis(hypo['pred_txt'], word_aligns))
        results.append(Hypotheses(hypotheses))
        
    return results


def main(args):
    # create model + decoder pair (change manually).
    if args.model_path:
        asr = Wav2Vec2_Decoder_Factory.get_cfg_beamsearch_fairseq(model_filepath=args.model_path, vocab_path=args.vocab_path, num_hyps=args.num_hyps, time_aligns=args.time_aligns)
        # asr = Wav2Vec2_Decoder_Factory.get_cfg_beamsearchtransformerlm(model_filepath=args.model_path, vocab_path=args.vocab_path, num_hyps=args.num_hyps, time_aligns=args.time_aligns)
        # asr = Wav2Vec2_Decoder_Factory.get_cfg_beamsearchkenlm_fairseq(model_filepath=args.model_path, vocab_path=args.vocab_path, num_hyps=args.num_hyps, time_aligns=args.time_aligns)
        # asr = Wav2Vec2_Decoder_Factory.get_cfg_viterbi(model_filepath=args.model_path, vocab_path=args.vocab_path, num_hyps=args.num_hyps, time_aligns=args.time_aligns)
    else:
        # default to a torchaudio wav2vec2 model if not using a custom .pt checkpoint.
        asr = Wav2Vec2_Decoder_Factory.get_torchaudio_beamsearch(num_hyps=args.num_hyps, time_aligns=args.time_aligns)
        # asr = Wav2Vec2_Decoder_Factory.get_torchaudio_greedy(num_hyps=args.num_hyps, time_aligns=args.time_aligns)

    # get all wav files as strings.
    wav_paths = get_all_wavs(args.in_dir, args.path_filters)

    # run ASR inference and decode into predicted hypothesis transcripts.
    results = asr.infer(wav_paths, batch_size=args.batch_size)

    # format decoded ASR inference results to a common interface.
    results = format_w2v2_output(results, args.time_aligns, args.num_hyps, asr.decoder)

    # write results hypotheses to output json files.
    write_results(results, wav_paths, args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ASR inference using a wav2vec2 checkpoint file and a dataset of audio files. Saves hypothesis transcripts to a new output folder. If ground truth transcript files exist, compile and save them in one file, so script output files are ready to be processed by sclite to calculate WER stats.")
    parser.add_argument("--in_dir", type=str, required=True,
                        help="Path to an existing folder containing wav audio files, optionally with corresponding txt transcript files for the corresponding audio files.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Path to a new output folder to create, where results will be saved.")
    parser.add_argument("--model_path", type=str, default='',
                        help="Path of a finetuned wav2vec2 model's .pt file. If unspecified, by default the script will use WAV2VEC2_ASR_LARGE_LV60K_960H torchaudio w2v2 model.")
    parser.add_argument("--vocab_path", type=str, default='',
                        help="Path of the finetuned wav2vec2 model's vocabulary text file (usually saved as dict.ltr.txt) that was used during wav2vec2 finetuning.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Minibatch size for inference. defaults to 1 (sequential processing). NOTE: decoding is always done sequentially, only ASR inference can be batched.")
    parser.add_argument("--num_hyps", type=int, default=1,
                        help="The number of best hypotheses to be returned by beam search decoding (if using beam search decoder) for an audio file. Defaults to 1 (i.e. returns just the best hypothesis). NOTE: this does not change beam size of decoder!")
    parser.add_argument("--time_aligns", default=False, action='store_true',
                        help="Flag used to specify whether to save word-level time alignment information along with the transcript for the hypothesis/hypotheses for decoders that support it. Defaults to False if flag is not provided.")
    parser.add_argument("--path_filters", type=str, nargs='+', default='',
                        help="List of keywords to filter the paths to wav files in the --in_dir directory. Will filter out any wav files that have those keywords present anywhere in their absolute path.")

    # parse command line arguments.
    args = parser.parse_args()

    # check arg vals if in allowable range.
    if args.batch_size < 1:
        raise ValueError("'--batch_size' should be a value >= 1 !!!")
    if args.num_hyps < 1:
        raise ValueError("'--num_hyps' should be a value >= 1 !!!")

    # setup logging to both console and logfile.
    utils.setup_logging(args.out_dir, 'wav2vec2_infer_custom.log', console=True, filemode='w')

    p = utils.Profiler()
    p.start()

    main(args)

    p.stop()