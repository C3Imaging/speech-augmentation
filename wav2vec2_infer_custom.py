
import os
import argparse
from Tools import utils
from pathlib import Path
from Tools.decoding_utils import Wav2Vec2_Decoder_Factory


def main(args):
    # get all wav files
    wav_paths = list(Path(args.in_dir).glob("**/*.wav"))
    # get accompanying ground truth transcripts files
    gt_tr_paths = list(Path(args.in_dir).glob("**/*.txt"))
    wav_paths.sort()
    gt_tr_paths.sort()

    # wavs_train = list(Path(args.in_dir).glob("kids_train/*.wav"))
    # wavs_test = list(Path(args.in_dir).glob("kids_test/*.wav"))
    # trs_train = list(Path(args.in_dir).glob("kids_train/*.txt"))
    # trs_test = list(Path(args.in_dir).glob("kids_test/*.txt"))
    # wavs_train_ids = [str(f).split('/')[-1].split('.wav')[0].split('myst_')[-1] for f in wavs_train]
    # trs_train_ids = [str(f).split('/')[-1].split('.txt')[0].split('myst_')[-1] for f in trs_train]
    # missing_from_wavs = list(sorted(set(trs_train_ids) - set(wavs_train_ids)))

    assert [str(f).split('/')[-1].split('.wav')[0].split('myst_')[-1] for f in wav_paths] == [str(f).split('/')[-1].split('.txt')[0].split('myst_')[-1] for f in gt_tr_paths], "number of and order of must be the same for audio and text filenames."

    # create model + decoder pair

    # cfg wav2vec2 + beam search decoder with Transformer language model from fairseq
    # asr = Wav2Vec2_Decoder_Factory.get_cfg_beamsearchtransformerlm(model_filepath=args.model_path, vocab_path=args.vocab_path)
    asr = Wav2Vec2_Decoder_Factory.get_cfg_viterbi(model_filepath=args.model_path, vocab_path=args.vocab_path)
    # run ASR inference and decode into predicted hypothesis transcripts
    hypos = asr.infer(wav_paths, batch_size=args.batch_size)
    # populate hypothesis.txt
    with open(os.path.join(args.out_dir, "hypothesis.txt"), 'w') as hyp_file:
        for hyp, wav_path in zip(hypos, wav_paths):
            # get unique id of audio sample
            id = str(wav_path).split('/')[-1].split('.wav')[0].split('myst_')[-1]
            hyp_file.write(f"{hyp} ({id})\n")
    # if there are ground truth transcripts accompanying the audio files, populate reference.txt
    if len(gt_tr_paths):
        with open(os.path.join(args.out_dir, "reference.txt"), 'w') as ref_file:
            for gt_tr_path in gt_tr_paths:
                # get unique id of transcript file
                id = str(gt_tr_path).split('/')[-1].split('.txt')[0].split('myst_')[-1]
                f = open(gt_tr_path, "r")
                tr = f.read().strip()
                f.close()
                ref_file.write(f"{tr} ({id})\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run speaker encoder on Librispeech dataset and create new folder with speakers that have highest cosine similarity in embeddings space compared to CMU kids speakers.")
    parser.add_argument("--in_dir", type=str, required=True,
                        help="Path to an existing folder containing wav audio files, optionally with corresponding txt transcript files for the corresponding audio files.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Path to a new output folder to create that will contain a file (hypothesis.txt) holding the generated transcripts and optionally a ground truth transcripts file (reference.txt)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path of a finetuned wav2vec2 model's .pt file")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path of the finetuned wav2vec2 model's vocab dict text file used during ASR finetuning")
    parser.add_argument("--batch_size", type=int,
                        help="Batch size for minibatches for inference. default 0 = sequential processing")

    # parse command line arguments
    args = parser.parse_args()

    # setup logging to both console and logfile
    utils.setup_logging(args.out_dir, 'wav2vec2_infer_custom.log', console=True, filemode='w')

    p = utils.Profiler()
    p.start()

    main(args)

    p.stop()