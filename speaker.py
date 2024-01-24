from pyannote.audio import Pipeline
# send pipeline to GPU (when available)
import torch
pipeline.to(torch.device("cuda"))

from split import *

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

class SpeakerDiarizator:

    def __init__(self, file):
        self.file = file
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE")
        self.audio_pipeline = pipeline(self.file)
