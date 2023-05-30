import os
import librosa
import logging
import numpy as np
from scipy.io.wavfile import write
from resemblyzer import preprocess_wav, VoiceEncoder


# constants
SAMPLING_RATE = 16000


def create_speaker_wavs(wav_path, speaker_names, start_stop_times):
    """Creates wav files per speaker, containing example speech from which speaker embeddings will be created by the Encoder model.
    If you want to create a 'global-speaker-samples/' dir, just rename the resultant created folder.
    
    Args:
      wav_path (str):
        path to the multispeaker wav file from which to extract speaker wavs.
      speaker_names (list str):
        explicit list of speaker names in order.
      start_stop_times (list float):
        list of pairs of start-stop times per speaker as a 1D array.
    """
    # create a subfolder for that wav audio file
    # create a separate subfolder for the diarization results for each audio file
    subfolder = os.path.join('/'.join(wav_path.split("/")[:-1]), "speaker-samples", wav_path.split("/")[-1].split(".wav")[0])
    if not os.path.exists(subfolder): os.makedirs(subfolder, exist_ok=True)
    wav, sr = librosa.load(wav_path, sr=None)
    if sr != SAMPLING_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLING_RATE)
    # Cut some segments from single speakers as reference audio.
    # Encoder model will later create speaker embeddings from manually selected segments of example speech from each speaker from a single audio file.
     # a segment representing a speaker = [start time in seconds, end time in seconds]
    for speaker_name, times in zip(speaker_names, [start_stop_times[i:i + 2] for i in range(0, len(start_stop_times), 2)]):
        speaker_wav = wav[int(times[0] * SAMPLING_RATE):int(times[1] * SAMPLING_RATE)]
        write(os.path.join(subfolder, f"{speaker_name}.wav"), SAMPLING_RATE, speaker_wav)


def get_speaker_samples(root_path):
    """Returns example wav per speaker from which speaker embeddings will be made by Voice Encoder model.
    NOTE: must manually populate 'root_path' dir with '<speakerID>.wav' audio files that contain example speech per speaker, from which speaker embeddings will be created. This can be created with the 'diarization/create_speaker_wavs.py' script.

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
    assert os.path.exists(root_path), f"ERROR: {root_path} does not exist!!! Please manually create it."

    speaker_names = []
    speaker_wavs = []
    for dirpath, _, filenames in os.walk(root_path, topdown=True):
        # get list of speech files from a single folder
        speech_files = []
        # loop through all files found
        for filename in filenames:
            if filename.endswith('.wav'):
                speech_files.append(os.path.join(dirpath, filename))
        break # loop only through files in topmost folder
    # loop through all files found
    assert len(speech_files), f"ERROR: no wav files found in {root_path}!!! Please manually add example speech wav files for each speaker."
    for speech_file in speech_files:
        # sr, wav = wavfile.read(os.path.join(dirpath, speech_file))
        wav, sr = librosa.load(os.path.join(dirpath, speech_file), sr=None)
        if sr != SAMPLING_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLING_RATE)
        speaker_wavs.append(wav)
        speaker_name = speech_file.split(".wav")[0].split("/")[-1]
        speaker_names.append(speaker_name)
    return speaker_names, speaker_wavs


def get_predictions(encoder, wav, speaker_names, speaker_embeds):
    """Voice Encoder model calculates speaker embeddings and similarity score predictions per speaker for each segment of audio.

    Args:
      encoder: (VoiceEncoder):
        the speaker encoder model object.
      wav (list, [int, float]):
        a multispeaker audio wav.
      speaker_names (list, str):
        speaker IDs.
      speaker_embeds (list, ndarray[float]):
        a fixed-length speaker embedding vector for each speaker.
    
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
    logging.info(f"Creating embedding for entire wav file.")
    # dividing wav into segments of multiple frames every 0.0625 seconds.
    # each segment will have an embedding of length 256 (same length as speaker embeddings).
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
    # pad the wav file with zeros, since embed_utterance() did for its segments.
    if wav_splits[-1].stop > len(wav) - 1:
        wav = np.pad(wav, (0, wav_splits[-1].stop - len(wav)), "constant")
    # Get the continuous similarity for every speaker. It amounts to a dot product between the 
    # embedding of the speaker and the continuous embedding of the interview.
    # for each speaker, for each segment, a similarity score value will be produced in the range [0.0,1.0]
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
        speaker_segments.append(name) if similarity > similarity_threshold else speaker_segments.append('none')
    return speaker_segments


def resemblyzer_diarization(root_path, similarity_threshold=0.7, global_speaker_embeds=False):
    """Main function to create Resemblyzer diarization output.
    NOTE: 'root_path' dir must contain a 'global-speaker-samples/' subdir that contains '<speakerID>.wav audio files, one per speaker, from which speaker embeddings will be created IF global_speaker_embeds=True. Otherwise, """

    encoder = VoiceEncoder("cpu")

    if global_speaker_embeds:
        speaker_names, speaker_wavs = get_speaker_samples(os.path.join(root_path, "speaker-samples", "global-speaker-samples"))
        # Get speaker embeddings
        speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
        logging.info("Global speaker embeddings created.")

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
        logging.info(f"Populating {subfolder} with diarization wavs based on {speech_file}.")

        # preprocesses audio into 16kHz mono-channel, removes long silences, normalises audio volume
        wav = preprocess_wav(speech_file)

        if global_speaker_embeds:
            wav, similarity_dict, wav_splits = get_predictions(encoder, wav, speaker_names, speaker_embeds)
        else:
            speaker_folder = os.path.join(root_path, "speaker-samples", speech_file.split("/")[-1].split(".wav")[0])
            speaker_names, speaker_wavs = get_speaker_samples(speaker_folder)
            # Get speaker embeddings
            speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
            wav, similarity_dict, wav_splits = get_predictions(encoder, wav, speaker_names, speaker_embeds)

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
            if v:
                speakers_frames_idxs_dict[k] = np.unique(np.concatenate(v).ravel())
                out_wav = os.path.join(subfolder, f"{k}_unified_confthresh_{str(similarity_threshold)}.wav")
                write(out_wav, SAMPLING_RATE, wav[speakers_frames_idxs_dict[k].tolist()])
                logging.info(f"{out_wav} created.")
            else:
                logging.info(f"{subfolder} has no diarized audio for speaker {k}.")
        logging.info(f"Population of {subfolder} with diarization wavs complete.")

