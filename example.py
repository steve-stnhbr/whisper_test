import torch
from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor, pipeline
import transformers
import math
from pyannote.audio import Pipeline
from split import Splitter
from os import listdir, remove
import sys
from os.path import isfile, join, basename
from tqdm.auto import tqdm
import pandas as pd
from optparse import OptionParser
import torchaudio
from pyannote.audio.pipelines.utils.hook import ProgressHook

def main(args):
    parser = OptionParser()
    parser.add_option("-d", "--diarization", dest="diarization", help="Diarization model to use", metavar="DIA_FILE")
    parser.add_option("-i", "--input", dest="input", help="Input file to transcribe", metavar="INPUT_FILE")
    parser.add_option("-o", "--output", dest="output", help="Output file to write to", metavar="OUTPUT_FILE")
    parser.add_option("-m", "--model", dest="model", help="ASR model to use", metavar="ASR_MODEL", default="openai/whisper-large-v3")
    parser.add_option("-s", "--speakers", dest="speakers", help="Number of speakers", metavar="SPEAKERS", default=2)

    (options, args) = parser.parse_args(args)

    if options.input is None:
        if len(args) == 0:
            print("No input file specified")
            exit(1)
        options.input = args[0]
    
    if options.output is None:
        options.output = f"out/{basename(options.input)}.txt"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print("Using device:", device)
    print("Using dtype:", torch_dtype)

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = WhisperProcessor.from_pretrained(model_id)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="german", task="transcribe")

    pipe = transformers.pipeline(
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

    if isfile(options.input):
        files = [options.input]
    else:
        files = [join(options.input, f) for f in listdir(options.input) if isfile(join(options.input, f))]
    if options.diarization is not None and isfile(options.diarization):
        diarization = pd.read_csv(options.diarization)
    else:
        diarization = None

    for file in files:
        print(f"Transcribing {file}")
        splitter = Splitter(file, "inter")
        print(f"Diarizing {file}")

        #aveform = splitter.audio
        #sample_rate = splitter.audio.frame_rate
        print("Loading audio from", file)
        waveform, sample_rate = torchaudio.load(file)

        if diarization is None:
            with ProgressHook() as hook:
                diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook, num_speakers=int(options.speakers))
            
            print("Finished diarization")
            # store the results in a file
            df_dia = pd.DataFrame(columns=['start', 'stop', 'speaker'])
            for speech_turn, track, speaker in diarization.itertracks(yield_label=True):
                df_dia.loc[-1] = [speech_turn.start, speech_turn.end, speaker]
                df_dia.index = df_dia.index + 1
                df_dia = df_dia.sort_index()
            df_dia.to_csv(f"out/{basename(file)}_diarization.csv", index=False)
            diarization = df_dia
            print(f"Finished diarizing {file}")
        else:
            print(f"Using diarization from {options.diarization}")

        speaker_before = None
        with open(f"out/{basename(file)}.txt", 'a') as f:
            for row in tqdm(diarization.iterrows(index=False), desc="ASR"):
                speaker_before = process_dia(pipe, row['start'], row['stop'], row['speaker'], speaker_before, f, splitter)
        print(f"Finished transcribing {file}, deleting inters")

        for inter in listdir("inter"):
            file = join("inter", inter)
            if isfile(file):
                remove(file) 

def process_dia(pipe, start, end, speaker, speaker_before, f, splitter):
    intermediary_file = splitter.single_split(start, end)
    result = pipe(intermediary_file)

    print(f"{speaker}:{result['text']}")
    if speaker_before != speaker:
        f.write(f"{speaker}:\n")
        speaker_before = speaker
    f.write(result["text"])
    f.write("\n")
    return speaker_before

if __name__ == "__main__":
    main(sys.argv[1:])