"""Simple script for processing the results of MOS evaluation Excel file with multiple sheets, where each sheet is a speaker ID."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

if __name__ == "__main__":
    # read Excel file with multiple sheets as a dataframe
    df = pd.read_excel('./PerSpeaker1.xlsx', sheet_name=None)
    # initialise a dict of speakers
    speakers_info = defaultdict(list)
    for speaker_id in df.keys():
        # create empty list as the value for a speaker key
        # each speaker key will map to a list, where el0 = mode of Q1, el1 = mean of Q2, el2 = mean of Q3
        speakers_info[speaker_id]

        # v1 statistics

        # calculate mode of the first column and add it as the first item in the list for that speaker
        col1_freqs = Counter(list(df[speaker_id].iloc[:,0]))
        mode = col1_freqs.most_common(1)[0][0]
        speakers_info[speaker_id].append(mode)
        # calculate mean of second and third columns and append to the list for that speaker
        col2 = list(df[speaker_id].iloc[:,1])
        mean_col2 = sum(col2) / len(col2)
        col3 = list(df[speaker_id].iloc[:,2])
        mean_col3 = sum(col3) / len(col3)
        speakers_info[speaker_id].append(mean_col2)
        speakers_info[speaker_id].append(mean_col3)

        # v2 statistics

        # calculate % of respondents that selected the mode for Q1 per speaker
        # and append to the list for that speaker
        mode_confidence_perc = 100 * col1_freqs[mode] / sum(col1_freqs.values())
        speakers_info[speaker_id].append(mode_confidence_perc)

        # bar plot of gender guess per speaker and save it as a png file
        gender_freqs = Counter(list(df[speaker_id].iloc[:,3]))
        labels, values = zip(*gender_freqs.items())
        indexes = np.arange(len(labels))

        plt.style.use('ggplot')
        fig, ax = plt.subplots()
        bars = ax.bar(indexes, values, color='blue')
        ax.bar_label(bars)
        for bars in ax.containers:
            ax.bar_label(bars)

        x_pos = [i for i, _ in enumerate(labels)]
        plt.xlabel("Gender guess")
        plt.ylabel("Frequency")
        plt.title(f"Gender guesses for speaker ID {speaker_id}")
        plt.xticks(x_pos, labels)
        for i, v in enumerate(values):
            plt.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
        plt.savefig(f"q4_speaker{speaker_id}.png")

    # create dataframe from the speaker info dict and save to csv file
    speaker_info_df = pd.DataFrame(speakers_info)
    speaker_info_df.index = ['Q1 Mode', 'Q2 Mean', 'Q3 Mean', 'Q1 Mode Confidence %'] # add row names
    speaker_info_df.to_csv('./PerSpeakerOutNUIG_Mariam.csv')
