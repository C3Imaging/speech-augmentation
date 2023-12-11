
import os
import json
import argparse
from Tools import utils
from pathlib import Path
from Tools.decoding_utils import Wav2Vec2_Decoder_Factory, normalize_timestamp_output, ViterbiDecoder, GreedyDecoder


def main(args):
    # get all wav files as strings.
    wav_paths = list(map(lambda x: str(x), list(Path(args.in_dir).glob("**/*.wav"))))
    # filter out any undesired wav files.
    if args.path_filters:
        for fil in args.path_filters:
            wav_paths = [w for w in wav_paths if fil not in w]
    
    # get accompanying ground truth transcripts files, if they exist.
    gt_tr_paths = list(map(lambda x: str(x), list(Path(args.in_dir).glob("**/*.txt"))))
    # filter out vocab file, hypothesis.txt file, reference.txt and any alignments.txt files.
    gt_tr_paths = list(filter(lambda x: not ("dict" in x or "hypothesis" in x or "hypotheses" in x or "reference" in x or "alignment" in x), gt_tr_paths))
    # filter out any undesired wav files' corresponding transcripts files.
    if args.path_filters:
        for fil in args.path_filters:
            gt_tr_paths = [t for t in gt_tr_paths if fil not in t]
    wav_paths.sort()
    gt_tr_paths.sort()

    # if there are ground truth transcript files:
    if gt_tr_paths:
        assert [f.split('/')[-1].split('.wav')[0].split('myst_')[-1] for f in wav_paths] == [f.split('/')[-1].split('.txt')[0].split('myst_')[-1] for f in gt_tr_paths], "number of and order of must be the same for audio and text filenames."

    # create model + decoder pair (change manually).
    if args.model_path:
        # asr = Wav2Vec2_Decoder_Factory.get_cfg_beamsearchkenlm_fairseq(model_filepath=args.model_path, vocab_path=args.vocab_path, num_hyps=args.num_hyps, time_aligns=args.time_aligns)
        asr = Wav2Vec2_Decoder_Factory.get_cfg_beamsearchtransformerlm(model_filepath=args.model_path, vocab_path=args.vocab_path, num_hyps=args.num_hyps, time_aligns=args.time_aligns)
        # asr = Wav2Vec2_Decoder_Factory.get_cfg_viterbi(model_filepath=args.model_path, vocab_path=args.vocab_path, num_hyps=args.num_hyps, time_aligns=args.time_aligns)
    else:
        # default to a torchaudio wav2vec2 model if not using a custom .pt checkpoint.
        asr = Wav2Vec2_Decoder_Factory.get_torchaudio_beamsearch(num_hyps=args.num_hyps, time_aligns=args.time_aligns)
        # asr = Wav2Vec2_Decoder_Factory.get_torchaudio_greedy(num_hyps=args.num_hyps, time_aligns=args.time_aligns)

    # run ASR inference and decode into predicted hypothesis transcripts.
    hypos = asr.infer(wav_paths, batch_size=args.batch_size)

    # populate best_hypotheses.json file (if num_hyps=1) or hypothesesX_of_N.json files (if num_hyps > 1).
    # GreedyDecoder and ViterbiDecoder ignore args.num_hyps.
    # GreedyDecoder ignores args.time_aligns.
    if args.num_hyps > 1 and not isinstance(asr.decoder, (ViterbiDecoder, GreedyDecoder)):
         # save all hypotheses to files.
        for all_hyps, wav_path in zip(hypos, wav_paths):
            for i in range(len(all_hyps)):
                with open(os.path.join(args.out_dir, f"hypotheses{i+1}_of_{len(all_hyps)}.json"), 'a') as hyp_file:
                    # create unique id of audio sample by including leaf folder in the id.
                    temp = wav_path.split('/')[-2:] # [0] = subfolder, [1] = ____.wav
                    temp[-1] = temp[-1].split('.wav')[0] # remove '.wav'
                    id = '/'.join(temp)
                    item = dict()
                    item['wav_path'] = wav_path
                    item['id'] = id
                    item['pred_txt'] = all_hyps[i]['pred_txt']
                    if args.time_aligns:
                        item['timestamps_word'] =  all_hyps[i]['timestamps_word']
                    hyp_file.write(json.dumps(item) + "\n")
    else:
        with open(os.path.join(args.out_dir, "best_hypotheses.json"), 'w') as hyp_file:
            for hyp, wav_path in zip(hypos, wav_paths):
                # create unique id of audio sample by including leaf folder in the id.
                temp = wav_path.split('/')[-2:] # [0] = subfolder, [1] = ____.wav
                temp[-1] = temp[-1].split('.wav')[0] # remove '.wav'
                id = '/'.join(temp)
                item = dict()
                item['wav_path'] = wav_path
                item['id'] = id
                item['pred_txt'] = hyp['pred_txt']
                if args.time_aligns and not isinstance(asr.decoder, GreedyDecoder):
                    item['timestamps_word'] = hyp['timestamps_word']
                hyp_file.write(json.dumps(item) + "\n")

    # if there are ground truth transcripts accompanying the audio files, populate reference.txt
    if len(gt_tr_paths):
        with open(os.path.join(args.out_dir, "reference.txt"), 'w') as ref_file:
            for gt_tr_path in gt_tr_paths:
                # create unique id of audio sample by including leaf folder in the id
                temp = gt_tr_path.split('/')[-2:] # [0] = subfolder, [1] = ____.txt
                temp[-1] = temp[-1].split('.txt')[0] # remove '.txt'
                id = '/'.join(temp)
                f = open(gt_tr_path, "r")
                tr = f.read().strip().lower()
                f.close()
                ref_file.write(f"{tr} ({id})\n")


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
                        help="Flag used to specify whether to save word-level time alignment information along with the transcript for the hypothesis/hypotheses. Defaults to False if flag is not provided.")
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