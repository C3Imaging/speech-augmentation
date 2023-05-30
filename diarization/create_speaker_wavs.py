import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, current + '/..')
import argparse
import resemblyzer_diarization_utils


def run_args_checks(args):
    assert len(args.times) % 2 == 0, "ERROR: please ensure each speaker has one start time and one stop time."
    assert len(args.times) == 2*len(args.speaker_names), "ERROR: please ensure each speaker has one start time and one stop time."
    for i in range(0, len(args.times), 2):
        start, stop = args.times[i:i + 2]
        assert start < stop, "ERROR: please ensure start time is less than stop time for each speaker."


def main(args):
    resemblyzer_diarization_utils.create_speaker_wavs(args.wav_path, args.speaker_names, args.times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates example wavs of speakers speech segments. Will create a 'speaker-samples/' subfolder under the dir the wav provided by --wav_path is in, and then a subsequent subfolder with the name being the wav's filename.")
    parser.add_argument("--wav_path", type=str, required=True,
                        help="Path to a wav file from which speaker embeddings will be extracted.")
    parser.add_argument("--speaker_names", type=str, nargs='+', required=True,
                        help="list of speaker names, e.g. --speaker_names adult child")
    parser.add_argument("--times", type=float, nargs='+', required=True,
                        help="list of start-stop times from the original audio per speaker. Times MUST be given in the same order as the speakers in the --speaker_names arg, and the order MUST be 'start_time' followed by 'stop_time' for each speaker. e.g. --times 2.5 5.2 18.7 21.2")
    args = parser.parse_args()
    run_args_checks(args)

    main(args)
