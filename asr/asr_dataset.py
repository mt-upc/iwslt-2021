import pandas as pd
import os
from tqdm import tqdm
from torch.utils import data
import torchaudio
import string
from num2words import num2words
import soundfile

torchaudio.set_audio_backend("soundfile")

class AsrDataset(data.Dataset):
	"""
	Dataset object for loading a datapoint, processing it,
	and passing to the dataloader
	"""

	def __init__(self, dir_path, tsv_file_name):
		"""
		Args:
			dir_path (str): path to the directory contains the tsv file
			tsv_file_name (str): name of the tsv file
		"""
		super(AsrDataset, self).__init__()

		self.dir_path = dir_path

		# dataframe
		self.data = pd.read_csv(os.path.join(dir_path, tsv_file_name), sep = "\t")

		# for cleaning target text
		self.to_remove = ["(Applause)", "(Laughter)", "(Applause ends)",
		"(Laughs)", "(Laughter ends)", "/u2014", "—"]
		self.my_punctuation = string.punctuation.replace("'", "")
		
	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		"""
		Args:
			i (int): index of the example in self.data
		Returns:
			Id (str): unique id of the example
			wav_array (1d np.array): raw audio frame data
			tgt_text (str): cleaned target text of the example
		"""

		# get datapoint
		Id, audio, tgt_text = self.data.iloc[i][["id", "audio", "tgt_text"]]

		# load audio frame into 1-d numpy array
		wav_file_path, offset, num_frames = audio.rsplit(":", 2)
		wav_array = torchaudio.load(os.path.join(self.dir_path, "data", wav_file_path),
			int(offset),
			int(num_frames))[0].numpy().squeeze(0)

		# clean target text
		tgt_text = self.clean(tgt_text)

		return Id, wav_array, tgt_text

	def clean(self, tgt_text):
		"""
		Produces a cleaned-version of the target text

		Args:
			tgt_text (str): target text of the example
		Returns:
			tgt_text (str): cleaned target text of the example
		"""

		if ": " in tgt_text:
			start_text, rest_text = tgt_text.split(": ", maxsplit = 1)
			start_tokens = start_text.split(" ")

			if len(start_tokens) == 1:
				if start_tokens[0].isupper():
					tgt_text = rest_text

			elif len(start_tokens) == 2:
				if start_tokens[0][0].isupper() and start_tokens[1][0].isupper():
					tgt_text = rest_text

		# remove non-words
		for seq in self.to_remove:
			tgt_text = tgt_text.replace(seq, "")

		# plus signs
		tgt_text = tgt_text.replace("+", "plus ")

		# remove punctuation, except apostrophes
		tgt_text = tgt_text.translate(str.maketrans("", "", self.my_punctuation))

		# trasform numbers to words
		tgt_text = " ".join([num2words(token).replace("and ", "") if token.isnumeric() else token
			for token in tgt_text.split(" ")])

		return tgt_text

