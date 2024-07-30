from pathlib import Path
from functools import partial, wraps

from beartype import beartype
from beartype.typing import Tuple, Union, Optional
from beartype.door import is_bearable

import torchaudio
from torchaudio.functional import resample

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import random
from utils import curtail_to_multiple
from scipy.io.wavfile import read
from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# type

OptionalIntOrTupleInt = Optional[Union[int, Tuple[Optional[int], ...]]]
MAX_WAV_VALUE = 32768.0
# dataset functions

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate
@beartype
class SoundDataset(Dataset):
    def __init__(
        self,
        # folder,
        training_files,
        split,
        segment_size,
        shuffle,
        validate=False,
        hop_length=80,
        exts = ['flac', 'wav'],
        max_length: OptionalIntOrTupleInt = None,
        target_sample_hz: OptionalIntOrTupleInt = None,
        seq_len_multiple_of: OptionalIntOrTupleInt = None
    ):
        super().__init__()
        # path = Path(folder)
        # assert path.exists(), 'folder does not exist'

        # files = [file for ext in exts for file in path.glob(f'**/*.{ext}')]
        # assert len(files) > 0, 'no sound files found'

        # self.files = files
        self.files = training_files

        self.split = split

        self.validate = validate
        self.hop_length = hop_length

        self.shuffle = shuffle
        self.segment_size = segment_size

        self.target_sample_hz = cast_tuple(target_sample_hz)
        num_outputs = len(self.target_sample_hz)

        self.max_length = cast_tuple(max_length, num_outputs)
        self.seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)

        assert len(self.max_length) == len(self.target_sample_hz) == len(self.seq_len_multiple_of)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        random.seed(1234)
        if self.shuffle:
            random.shuffle(self.files)
        file = self.files[idx]

        data, sample_hz = load_wav(file)

        data = data / MAX_WAV_VALUE

        data = torch.FloatTensor(data)

        assert data.numel() > 0, f'one of your audio file ({file}) is empty. please remove it from your folder'

        data = data.unsqueeze(0)


        if self.split:
            if data.size(1) >= self.segment_size:
                max_start = data.size(1) - self.segment_size
                data_start = random.randint(0, max_start)
                data = data[:, data_start:data_start + self.segment_size]
            else:
                data = torch.nn.functional.pad(data, (0, self.segment_size - data.size(1)), 'constant')
        
                
        return data.squeeze(0)


        

# dataloader functions

def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = torch.stack(data)
            return (data,)

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return tuple(outputs)

    return inner

@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first = True)

def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)
