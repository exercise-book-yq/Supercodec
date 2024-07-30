from __future__ import absolute_import, division, print_function, unicode_literals
from re import T
import numpy as np
import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from utils import AttrDict
from utils import mel_spectrogram, MAX_WAV_VALUE, load_wav, get_dataset_filelist
from models import Generator
from supercodec_causal import Supercodec
from data import SoundDataset, get_dataloader
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm
h = None
device = None
# import Time

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    print(h.rq_num_quantizers)

    soundstream = Supercodec(
        codebook_size=h.codebook_size,
        codebook_dim=h.codebook_dim,
        rq_num_quantizers=h.rq_num_quantizers,
        shared_codebook = False,
        strides=h.strides,
        channel_mults=h.channel_mults,
        training=False
        ).cuda()
    
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    soundstream.load_state_dict(state_dict_g['generator'])


    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    soundstream.eval()
    for filepath in tqdm(os.listdir(a.input_wavs_dir)):
        filelist = os.listdir(a.input_wavs_dir+filepath)
        filelist.sort()

        if not os.path.exists(a.output_dir+filepath+'/'):
            os.makedirs(a.output_dir+filepath+'/')

        
        with torch.no_grad():
            for i, filename in enumerate(filelist):
                wave, sr = load_wav(os.path.join(a.input_wavs_dir+filepath, filename))
                wave = wave / MAX_WAV_VALUE
                wave = torch.FloatTensor(wave)

                wave = wave.to(device)
                y_g_hat = soundstream(wave, return_recons_only = True)
                
                audio = y_g_hat.squeeze()
                audio = audio * MAX_WAV_VALUE
                audio = audio.cpu().numpy().astype('int16')

                output_file = os.path.join(a.output_dir+filepath+'/', os.path.splitext(filename)[0] + '.wav')
                write(output_file, h.sampling_rate, audio)



def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default="")
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--checkpoint_file', default='')
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

