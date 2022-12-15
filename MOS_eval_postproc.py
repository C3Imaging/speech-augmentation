"""Simple script for processing the results of MOS evaluation Excel file with multiple sheets, where each sheet is a speaker ID."""

import pandas as pd
from collections import defaultdict, Counter

if __name__ == "__main__":
    # read Excel file with multiple sheets as a dataframe
    df = pd.read_excel('./PerSpeaker.xlsx', sheet_name=None)
    # initialise a dict of speakers
    speakers_info = defaultdict(list)
    for speaker_id in df.keys():
        # create empty list as the value for a speaker key
        # each speaker key will map to a list, where el0 = mode of Q1, el1 = mean of Q2, el2 = mean of Q3
        speakers_info[speaker_id]
        # calculate mode of the first column and add it as the first item in the list for that speaker
        col1_freqs = Counter(list(df[speaker_id].iloc[:,0]))
        speakers_info[speaker_id].append(col1_freqs.most_common(1)[0][0])
        # calculate mean of second and third columns and append to the list for that speaker
        col2 = list(df[speaker_id].iloc[:,1])
        mean_col2 = sum(col2) / len(col2)
        col3 = list(df[speaker_id].iloc[:,2])
        mean_col3 = sum(col3) / len(col3)
        speakers_info[speaker_id].append(mean_col2)
        speakers_info[speaker_id].append(mean_col3)
    # create dataframe from the speaker info dict and save to csv file
    speaker_info_df = pd.DataFrame(speakers_info)
    speaker_info_df.to_csv('./PerSpeakerOut.csv')