import torch
from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor, pipeline
import math
from pyannote.audio import Pipeline
from split import Splitter
from os import listdir, remove
from os.path import isfile, join, basename
from tqdm.auto import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = WhisperProcessor.from_pretrained(model_id)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="german", task="transcribe")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")
pipeline.to(torch.device(device))

files = [join("data", f) for f in listdir("data") if isfile(join("data", f))]


for file in files:
    print(f"Transcribing {file}")
    splitter = Splitter(file, "inter")
    print(f"Diarizing {file}")
    diarization = pipeline(file)

    print("Finished diarization, now doing ASR")
    speaker_before = None
    with open(f"out/{basename(file)}.txt", 'a') as f:
        for turn, _, speaker in tqdm(diarization.itertracks(yield_label=True), desc="ASR"):
            print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            intermediary_file = splitter.single_split(turn.start, turn.end)
            result = pipe(intermediary_file)
        
            print(f"Speaker{speaker}:{result['text']}")
            if speaker_before != speaker:
                f.write(f"{speaker}:\n")
                speaker_before = speaker
            f.write(result["text"])
            f.write("\n")
    print(f"Finished transcribing {file}, deleting inters")

    for inter in listdir("inter"):
        file = join("inter", inter)
        if isfile(file):
            remove(file) 
