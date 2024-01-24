from pydub import AudioSegment
import math
from os.path import basename
import torchaudio

class Splitter():
    def __init__(self, filename, out_folder):
        print(f"Creating splitter for {filename}")
        self.filepath = filename
        self.out_folder = out_folder
        self.waveform, self.sample_rate = torchaudio.load(self.filepath)
        
        # if (self.filepath.endswith('.mp3')):
        #     self.audio = AudioSegment.from_mp3(self.filepath)
        # else:
        #     self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_s, to_s):
        print(f"Extracting {self.filepath} ({from_s}, {to_s})")
        t1 = from_s * self.sample_rate
        t2 = to_s * self.sample_rate
        split_audio = self.waveform[t1:t2]
        name = basename(self.filepath)
        file_name = self.out_folder + f'/{name}_split_{int(from_s)}_{int(to_s)}.wav'
        #split_audio.export(file_name, format="mp3")
        torchaudio.save(file_name, split_audio, self.sample_rate)
        return file_name
        
    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration() / 60)
        for i in range(0, total_mins, min_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+min_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_mins - min_per_split:
                print('All splited successfully')