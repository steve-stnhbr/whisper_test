import torch
from transformers import pipeline
import torchaudio


mp3_file_path = 'test.m4a'
print(f"loading file{mp3_file_path}")
# Load the audio file using torchaudio
waveform, sample_rate = torchaudio.load(mp3_file_path)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"loading model on {device}")

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-large-v2",
  chunk_length_s=30,
  device=device,
)

print(f"starting prediction")

# we can also return timestamps for the predictions
prediction = pipe(waveform.numpy(), batch_size=8, return_timestamps=True)["chunks"]
with open("out/output.txt", "w") as f:
  for p in prediction:
    f.write(p["text"])