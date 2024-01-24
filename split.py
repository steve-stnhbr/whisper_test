from pydub import AudioSegment
import math
from os.path import basename

class Splitter():
    def __init__(self, filename, out_folder):
        print(f"Creating splitter for {filename}")
        self.filepath = filename
        self.out_folder = out_folder
        
        if (self.filepath.endswith('.mp3')):
            self.audio = AudioSegment.from_mp3(self.filepath)
        else:
            self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_s, to_s):
        print(f"Extracting {self.filepath} ({from_s}, {to_s})")
        t1 = from_s * 1000
        t2 = to_s * 1000
        split_audio = self.audio[t1:t2]
        name = basename(self.filepath)
        file_name = self.out_folder + f'/{name}_split_{int(from_s)}_{int(to_s)}.mp3'
        split_audio.export(file_name, format="mp3")
        return file_name
        
    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration() / 60)
        for i in range(0, total_mins, min_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+min_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_mins - min_per_split:
                print('All splited successfully')