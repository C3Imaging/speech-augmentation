# run ASR inference using a wav2vec2 ASR model and a specified decoder on a single audio file.
# NOTE: this script can only use wav2vec2 ASR models from torchaudio library.

import os
import torch
import torchaudio
import torchaudio.models.decoder
from Tools import decoding_utils_torch

if __name__ == "__main__":
    # folder for saving output wavefiles (and manually plots if needed)
    new_dir = "WAV2VEC2_ASR_LARGE_LV60K_960H"
    new_dir = os.path.join("/workspace/projects/Alignment/wav2vec2_alignment/single_audio_outputs/decoding", new_dir)
    if not os.path.exists(new_dir): os.makedirs(new_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wav2vec2 model
    # bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
    acoustic_model = bundle.get_model().to(device)
    # acoustic model's vocabulary of chars
    labels =  [label.lower() for label in bundle.get_labels()]
    # KenLM Librispeech language model
    files = torchaudio.models.decoder.download_pretrained_files("librispeech-4-gram")

    # initialise greedy CTC decoder with no language model
    greedy_decoder = decoding_utils_torch.GreedyCTCDecoder(labels)

    # initialise beam search decoder with language model
    beam_search_decoder = torchaudio.models.decoder.ctc_decoder(
        lexicon=files.lexicon, # giant file of English "words"
        tokens=files.tokens, # same as wav2vec2's vocab
        lm=files.lm, # path to language model binary
        nbest=3,
        beam_size=1500,
        lm_weight=3.23,
        word_score=-0.26,
    )

    # Specify path to <audio, gt_transcript> files
    speech_file = r"/workspace/datasets/myst_test/myst_999465_2009-17-12_00-00-00_MS_4.2_024.wav"
    transcript_path = r"/workspace/datasets/myst_test/myst_999465_2009-17-12_00-00-00_MS_4.2_024.txt"
    # preprocess transcript file MYST
    with open(transcript_path, mode="r", encoding="utf-8") as f:
        transcript = f.read().lower().split(" ") # split transcript into a list of words

    with torch.inference_mode():
        # load audio file and resample if needed
        waveform, sample_rate = torchaudio.load(speech_file)
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        # inference
        emissions, _ = acoustic_model(waveform.to(device))
        #emissions = torch.log_softmax(emissions, dim=-1)

    # greedy decoding
    greedy_result = greedy_decoder(emissions) # decoded phrase as a list of words
    greedy_wer = torchaudio.functional.edit_distance(transcript, greedy_result) / len(transcript)

    # beam search decoding
    beam_search_result = beam_search_decoder(emissions.cpu().detach())[0][0].words # decoded phrase as a list of words
    beam_search_wer = torchaudio.functional.edit_distance(transcript, beam_search_result) / len(transcript)

    greedy_transcript = ' '.join(greedy_result)
    beam_search_transcript = ' '.join(beam_search_result)
    transcript = ' '.join(transcript)

    # get speech file ID
    file_id = speech_file.split("/")[-1].split(".wav")[0]
    # save resultant transcripts
    with open(os.path.join(new_dir, 'transcripts.txt'), 'w') as f:
        f.write(f"{file_id}\n") # time is in seconds
        f.write("\n")
        f.write("Transcript 1: ground truth transcript\n")
        f.write("Transcript 2: greedy CTC decoder transcript\n")
        f.write("Transcript 3: beam search decoder + KenLM + lexicon transcript\n")
        f.write("\n")
        f.write(f"greedy CTC decoder WER: {greedy_wer}\n")
        f.write(f"beam search + KenLM + lexicon WER: {beam_search_wer}\n")
        f.write("--------------------------------------------------------\n")
        f.write(transcript + "\n")
        f.write(greedy_transcript + "\n")
        f.write(beam_search_transcript + "\n")
