import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, current + '/..')
from Tools import utils
from diarization_utils import *

# import wave
# w1 = wave.open("/workspace/datasets/Wearable_Audio/test2/5098.wav")
# print("Number of channels is: ",    w1.getnchannels())
# print("Sample width in bytes is: ", w1.getsampwidth())
# print("Framerate is: ",             w1.getframerate())
# print("Number of frames is: ",      w1.getnframes())

def main(root_path):
    utils.mp3_to_wav(root_path)
    utils.preprocessing_augmentations(root_path, in_place=True)
    pyannote_diarization(root_path)
    rttm_to_wav(os.path.join(root_path, "pyannote-diarization"))


if __name__ == "__main__":
    root_path = "/workspace/datasets/Wearable_Audio/test2"
    main(root_path)
