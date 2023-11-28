import pandas as pd


def load_hypothesis(path: str) -> pd.DataFrame:
    """Returns a pandas Dataframe version of the hypothesis transcript.

        Args:
          path (str):
            The path to a hypothesis transcript in the common format with the following columns:
                [confidence_score (optional)], word_label, start_time, stop_time
        
        Returns:
          (pd.DataFrame): hypothesis transcript in pandas Dataframe format.
    """
    return pd.read_csv(path)


if __name__ == "__main__":
    path_test = "/workspace/datasets/Wearable_Audio_whisper_test/pyannote-diarization/3772/SPEAKER_03/SPEAKER_03/WHISPER_ALIGNS_whisper_alignments/3772_SPEAKER_03_1/alignments.txt"
    hyp_paths = [path_test]
    # [ [hyp1_sp1, hyp2_sp1, hyp3_sp1], [hyp1_sp2, hyp2_sp2, hyp3_sp3], ... ]

    for speaker in 
    load_hypothesis(path_test)