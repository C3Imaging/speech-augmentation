"""Quick script to test the time alignments produced by 'wav2vec2_infer_custom.py' by saving each word as a separate audio file."""

import json
import shlex
import subprocess
import os

root_dir = '/workspace/datasets/LibriTTS_test_whisper/w2v2_infer_out_fairseq_transformerlm_maxaudiolen_maxnumframes_batchsize2'
out_dir = os.path.join(root_dir, 'words')
if not os.path.exists(out_dir): os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(root_dir, 'hypotheses1_of_3.txt'), 'r', encoding='utf-8') as fr:
    for audio_idx, line in enumerate(fr):
        item = json.loads(line)
        for word_idx, word in enumerate(item['timestamps_word']):
            out_audio_path = os.path.join(out_dir, f"audio{audio_idx}_word{word_idx}_{word['word']}.wav")
            duration = word['end_time'] - word['start_time']
            subprocess.run(shlex.split(f"ffmpeg -y -ss {word['start_time']} -i {item['wav_path']} -t {duration} {out_audio_path}"))