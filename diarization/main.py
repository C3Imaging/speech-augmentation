import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, current + '/..')
from Tools import utils
import pyannote_diarization_utils
import resemblyzer_diarization_utils

# ------- debugging audio characteristics -------

# import wave
# w1 = wave.open("/workspace/datasets/Wearable_Audio/test2/5098.wav")
# print("Number of channels is: ",    w1.getnchannels())
# print("Sample width in bytes is: ", w1.getsampwidth())
# print("Framerate is: ",             w1.getframerate())
# print("Number of frames is: ",      w1.getnframes())

# OR

# from pydub import AudioSegment
# song = AudioSegment.from_file("/workspace/datasets/Wearable_Audio/test2/5098.wav", format="wav")
# song = AudioSegment.from_file("/workspace/datasets/test/yt.m4a")
# print(song.frame_rate)
# print(song.channels)
# -------------------------------------------------


def pyannote_pipeline(root_path):
    # pyannote pipeline
    utils.mp3_to_wav(root_path)
    utils.preprocessing_augmentations(root_path, in_place=True)
    pyannote_diarization_utils.pyannote_diarization(root_path)
    speaker_folders = pyannote_diarization_utils.rttm_to_wav(os.path.join(root_path, "pyannote-diarization"))
    for speaker_folder in speaker_folders:
        # call for each speaker subfolder
        pyannote_diarization_utils.combine_wavs(speaker_folder)
        pyannote_diarization_utils.combine_wavs(speaker_folder)


def main(root_path):
    # pyannote pipeline
    pyannote_pipeline(root_path)

    # resemblyzer pipeline
    resemblyzer_diarization_utils.resemblyzer_diarization(root_path, similarity_threshold=0.7)


if __name__ == "__main__":
    root_path = "/workspace/datasets/Wearable_Audio/test-both"
    
    main(root_path)
