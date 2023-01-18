
import os
import argparse
from Tools import utils
from pathlib import Path
from Tools.decoding_utils import Wav2Vec2_Decoder_Factory


def main(args):
    # get all wav files as strings
    wav_paths = list(map(lambda x: str(x), list(Path(args.in_dir).glob("**/*.wav"))))
    # get accompanying ground truth transcripts files
    gt_tr_paths = list(map(lambda x: str(x), list(Path(args.in_dir).glob("**/*.txt"))))
    # filter out vocab file, hypothesis.txt file, and reference.txt file
    gt_tr_paths = list(filter(lambda x: not ("dict" in x or "hypothesis" in x or "reference" in x), gt_tr_paths))
    wav_paths.sort()
    gt_tr_paths.sort()

    # # DEBUG CODE: find unmatching <wav,txt> files pairs
    # wavs_train = list(Path(args.in_dir).glob("kids_train/*.wav"))
    # wavs_test = list(Path(args.in_dir).glob("kids_test/*.wav"))
    # trs_train = list(Path(args.in_dir).glob("kids_train/*.txt"))
    # trs_test = list(Path(args.in_dir).glob("kids_test/*.txt"))
    # wavs_train_ids = [f.split('/')[-1].split('.wav')[0].split('myst_')[-1] for f in wavs_train]
    # trs_train_ids = [f.split('/')[-1].split('.txt')[0].split('myst_')[-1] for f in trs_train]
    # missing_from_wavs = list(sorted(set(trs_train_ids) - set(wavs_train_ids)))

    assert [f.split('/')[-1].split('.wav')[0].split('myst_')[-1] for f in wav_paths] == [f.split('/')[-1].split('.txt')[0].split('myst_')[-1] for f in gt_tr_paths], "number of and order of must be the same for audio and text filenames."

    # create model + decoder pair

    # cfg wav2vec2 + beam search decoder with Transformer language model from fairseq
    # asr = Wav2Vec2_Decoder_Factory.get_cfg_beamsearchtransformerlm(model_filepath=args.model_path, vocab_path=args.vocab_path)
    asr = Wav2Vec2_Decoder_Factory.get_cfg_viterbi(model_filepath=args.model_path, vocab_path=args.vocab_path)
    # run ASR inference and decode into predicted hypothesis transcripts
    hypos = asr.infer(wav_paths, batch_size=args.batch_size)
    # populate hypothesis.txt
    with open(os.path.join(args.out_dir, "hypothesis.txt"), 'w') as hyp_file:
        for hyp, wav_path in zip(hypos, wav_paths):
            # create unique id of audio sample by including leaf folder in the id
            temp = wav_path.split('/')[-2:] # [0] = subfolder, [1] = ____.wav
            temp[-1] = temp[-1].split('.wav')[0] # remove '.wav'
            id = '/'.join(temp)
            hyp_file.write(f"{hyp} ({id})\n")
    # if there are ground truth transcripts accompanying the audio files, populate reference.txt
    if len(gt_tr_paths):
        with open(os.path.join(args.out_dir, "reference.txt"), 'w') as ref_file:
            for gt_tr_path in gt_tr_paths:
                # get unique id of transcript file
                id = gt_tr_path.split('/')[-1].split('.txt')[0].split('myst_')[-1]
                f = open(gt_tr_path, "r")
                tr = f.read().strip()
                f.close()
                ref_file.write(f"{tr} ({id})\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ASR inference using a wav2vec2 checkpoint file and a dataset of audio files. Saves hypothesis transcripts to a new output folder.")
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