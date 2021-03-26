import pandas as pd
import os
from tqdm import tqdm
from torch.utils import data
import torchaudio

torchaudio.set_audio_backend("soundfile")

class AsrDataset(data.Dataset):
	def __init__(self, dir_path, tsv_file_name):
		super(AsrDataset, self).__init__()

		self.dir_path = dir_path
		self.data = pd.read_csv(os.path.join(dir_path, tsv_file_name), sep = "\t")
		
	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):

		Id, audio, tgt_text = self.data.iloc[i][["id", "audio", "tgt_text"]]

		wav_file_path, offset, num_frames = audio.rsplit(":", 2)
		wav_array = torchaudio.load(os.path.join(self.dir_path, "data", wav_file_path),
								int(offset),
								int(num_frames))[0].numpy().squeeze(0)

		return Id, wav_array, tgt_text