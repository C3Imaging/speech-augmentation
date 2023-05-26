import os
import librosa
import numpy as np
from scipy.io.wavfile import write
from resemblyzer import preprocess_wav, VoiceEncoder


# constants
SAMPLING_RATE = 16000


def get_speaker_samples(root_path):
    """Returns example wav per speaker from which speaker embeddings will be made by Voice Encoder model.
    NOTE: must manually populate a 'speaker_samples/' subdir, of the 'root_path' folder, with '<speakerID>.wav' audio files that contain example speech per speaker, from which speaker embeddings will be created.

    Args:
      root_path (str):
        The path to a folder contianing wav audio files and a 'speaker_samples/' subdir.
    
    Returns:
      speaker_names (list, str):
        speaker IDs.
      speaker_wavs (list, np.ndarray[int, float]):
        wav data per speaker.
    """
    # load example speech for speakers, from which speaker embeddings will be created
    speaker_names = []
    speaker_wavs = []
    for dirpath, _, filenames in os.walk(os.path.join(root_path, "speaker_samples"), topdown=True):
        # loop through all files found
        for filename in filenames:
            if filename.endswith('.wav'):
                # sr, wav = wavfile.read(os.path.join(dirpath, filename))
                wav, sr = librosa.load(os.path.join(dirpath, filename), sr=None)
                if sr != SAMPLING_RATE:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLING_RATE)
                speaker_wavs.append(wav)
                speaker_names.append(filename.split(".wav")[0])
        break
    return speaker_names, speaker_wavs


def get_predictions(wav, speaker_names, speaker_wavs):
    """Voice Encoder model calculates speaker embeddings and similarity score predictions per speaker for each segment of audio.

    Args:
      wav (list, [int, float]):
        A multispeaker audio wav.
      speaker_names (list, str):
        speaker IDs.
      speaker_wavs (list, list[int, float]):
        wav data per speaker.
    
    Returns:
      wav (list, [int, float]):
        A multispeaker audio wav, possibly padded.
      similarity_dict (dict, {str: np.ndarray}):
        keys are speaker IDs, values are similarity scores for each audio segment as cut by wav_splits.
      wav_splits (list, slice):
        start and stop frame indexes of each audio segment.
    """
    ## Compare speaker embeds to the continuous embedding of the interview
    # Derive a continuous embedding of the interview. We put a rate of 16, meaning that an 
    # embedding is generated every 0.0625 seconds. It is good to have a higher rate for speaker 
    # diarization, but it is not so useful for when you only need a summary embedding of the 
    # entire utterance. A rate of 2 would have been enough, but 16 is nice for the sake of the 
    # demonstration. 
    # We'll exceptionally force to run this on CPU, because it uses a lot of RAM and most GPUs 
    # won't have enough. There's a speed drawback, but it remains reasonable.
    encoder = VoiceEncoder("cpu")
    print("Running the continuous embedding on cpu, this might take a while...")
    # dividing wav into segments of multiple frames every 0.0625 seconds.
    # each segment will have an embedding on length 256.
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
    # pad the wav file with zeros, since embed_utterance() did for its segments.
    if wav_splits[-1].stop > len(wav) - 1:
        wav = np.pad(wav, (0, wav_splits[-1].stop - len(wav)), "constant")

    # Get the continuous similarity for every speaker. It amounts to a dot product between the 
    # embedding of the speaker and the continuous embedding of the interview.
    # for each speaker, for each segment, a similarity score value will be produced in the range [0.0,1.0]
    speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
    similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in 
                    zip(speaker_names, speaker_embeds)}
    
    return wav, similarity_dict, wav_splits


def get_speaker_segments(similarity_dict, wav_splits, similarity_threshold):
    """Returns a list of most likely speaker IDs per segment.

    Args:
      similarity_dict (dict, {str: np.ndarray}):
        keys are speaker IDs, values are similarity scores for each audio segment as cut by wav_splits.
      wav_splits (list, slice):
        start and stop frame indexes of each audio segment.
      similarity_threshold (float):
        minimum confidence threshold for considering a speech segment to be confidently spoken by a speaker.
    
    Returns:
      speaker_segments (list, str):
        most likely speaker ID per segment.
    """
    # a speaker per segment, 'none' if segment has similarity score below similarity_threshold for all known speakers.
    speaker_segments = list()
    # loop through each segment
    for i in range(len(wav_splits)):
        # get speaker with max similarity
        similarities = [s[i] for s in similarity_dict.values()]
        best = np.argmax(similarities)
        name, similarity = list(similarity_dict.keys())[best], similarities[best]
        if similarity > similarity_threshold:
            speaker_segments.append(name)
        else:
            speaker_segments.append('none')
    return speaker_segments


def resemblyzer_diarization(root_path, similarity_threshold):
    """Main function to create Resemblyzer diarization output.
    NOTE: 'root_path' dir must contain a 'speaker_samples/' subdir that contains '<speakerID>.wav audio files, one per speaker, from which speaker embeddings will be created."""

    speaker_names, speaker_wavs = get_speaker_samples(root_path)

    # # Cut some segments from single speakers as reference audio
    # creates speaker embeddings from manually selected segments of example speech from each speaker from a single audio file.
    # segments = [[2.5, 5.2], [18.7, 21.2]] # a segment representing a speaker = [start time in seconds, end time in seconds]
    # speaker_names = ["adult", "child"]
    # speaker_wavs = [wav[int(s[0] * SAMPLING_RATE):int(s[1] * SAMPLING_RATE)] for s in segments]
    # write('/workspace/datasets/Wearable_Audio/test3_resemblyzer/adult_sample.wav', SAMPLING_RATE, speaker_wavs[0])
    # write('/workspace/datasets/Wearable_Audio/test3_resemblyzer/child_sample.wav', SAMPLING_RATE, speaker_wavs[1])

    for dirpath, _, filenames in os.walk(root_path, topdown=True):
        # get list of speech files from a single folder
        speech_files = []
        # loop through all files found
        for filename in filenames:
            if filename.endswith('.wav'):
                speech_files.append(os.path.join(dirpath, filename))
        break # loop only through files in topmost folder

    # create new subfolder for the diarization results
    out_path = os.path.join(root_path, "resemblyzer-diarization")
    if not os.path.exists(out_path): os.makedirs(out_path, exist_ok=True)

    # main loop
    for speech_file in speech_files:
        # apply the pipeline to an audio file
        # create a separate subfolder for the diarization results for each audio file
        subfolder = os.path.join(out_path, speech_file.split("/")[-1].split(".wav")[0])
        if not os.path.exists(subfolder): os.makedirs(subfolder, exist_ok=True)

		# preprocesses audio into 16kHz mono-channel, removes long silences, normalises audio volume
        wav = preprocess_wav(speech_file)

        wav, similarity_dict, wav_splits = get_predictions(wav, speaker_names, speaker_wavs)

        speaker_segments = get_speaker_segments(similarity_dict, wav_splits, similarity_threshold)

        # initialise empty list per speaker.
        # elements of a list will be the frame indexes of the original audio file where that speaker has a confidence value above the thrseshold.
        speakers_frames_idxs_dict = dict()
        for speaker in speaker_names:
            speakers_frames_idxs_dict[speaker] = list()

        # add the indexes of frames (a list) determined to be the speaker speaking to the corresponding speaker key (will be a list of lists, where each lowest-level list is a segment of frame indexes)
        for speaker, seg in zip(speaker_segments, wav_splits):
            if speaker != 'none':
                speakers_frames_idxs_dict[speaker].append(np.arange(seg.start,seg.stop))
        # collapse each speaker's lists into a 1D array of sorted unique frame indexes and write the corresponding audio frames to separate speaker files.
        for k, v in speakers_frames_idxs_dict.items():
            speakers_frames_idxs_dict[k] = np.unique(np.concatenate(v).ravel())
            write(os.path.join(subfolder, f"{k}_unified_confthresh_{str(similarity_threshold)}.wav"), SAMPLING_RATE, wav[speakers_frames_idxs_dict[k].tolist()])
