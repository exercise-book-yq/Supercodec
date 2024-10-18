import functools
from itertools import cycle
from pathlib import Path
from tkinter.tix import Select
import numpy as np
from functools import partial, wraps
from itertools import zip_longest
from scipy import signal
import torch
from torch import nn, einsum
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from torch.linalg import vector_norm
import time
import torchaudio.transforms as T
from torchaudio.functional import resample
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from einops import rearrange, reduce, pack, unpackk

from vector_quantize_pytorch import ResidualVQ
# from residual_vq import ResidualVQ

from utils import curtail_to_multiple
from utils import init_weights
from version import __version__
from packaging import version

parsed_version = version.parse(__version__)

from modules.snaked import Snake1d
from modules.conv import pad1d, unpad1d, get_extra_padding_for_conv1d

import pickle


# helper functions
def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(t, l=1):
    return ((t,) * l) if not isinstance(t, tuple) else t


def filter_by_keys(fn, d):
    return {k: v for k, v in d.items() if fn(k)}


def map_keys(fn, d):
    return {fn(k): v for k, v in d.items()}


# gan losses

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


def hinge_gen_loss(fake):
    return -fake.mean()


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


LRELU_SLOPE = 0.1


def gradient_penalty(wave, output, weight=10):
    batch_size, device = wave.shape[0], wave.device

    gradients = torch_grad(
        outputs=output,
        inputs=wave,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((vector_norm(gradients, dim=1) - 1) ** 2).mean()


class SConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, causal, pad_mode='reflect', **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = kwargs.get('dilation', 1)
        self.stride = kwargs.get('stride', 1)
        self.pad_mode = pad_mode
        self.padding = self.dilation * (kernel_size - 1) + 1 - self.stride
        self.conv = weight_norm(nn.Conv1d(chan_in, chan_out, self.kernel_size, **kwargs))
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.causal = causal


    def forward(self, x):
        extra_padding = get_extra_padding_for_conv1d(x, self.kernel_size, self.stride, self.padding)
        if self.causal:
            x = pad1d(x, (self.padding, extra_padding), mode=self.pad_mode)
        else:
            padding_right = self.padding // 2
            padding_left = self.padding - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)
    
class SConvTranspose1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride, **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.padding = kernel_size - 1
        self.conv = weight_norm(nn.ConvTranspose1d(chan_in, chan_out, kernel_size, stride, **kwargs))

    def forward(self, x):
        n = x.shape[-1]

        out = self.conv(x)
        out = out[..., :(n * self.upsample_factor)]

        return out
    
class SelectNet(nn.Module):
    def __init__(
            self, 
            in_channels, 
            kernel_size=3, 
            M=2, 
            r=2, 
            stride=1, 
            L=32, 
            G=1, 
            causal=False, 
            pad_mode='reflect'):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SelectNet, self).__init__()
        d = max(int(in_channels / r), L)
        self.M = M
        self.features = in_channels

        self.fc = nn.Sequential(
            Snake1d(in_channels),
            SConv1d(chan_in=in_channels, chan_out=in_channels, kernel_size=kernel_size, causal=causal, pad_mode=pad_mode)
            )
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                SConv1d(chan_in=in_channels, chan_out=in_channels, kernel_size=kernel_size, causal=causal, pad_mode=pad_mode),
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, fea_U, s, n):

        out = torch.stack((s, n), dim=-1)
        fea_s = fea_U.mean(-1)
        fea_s = fea_s.unsqueeze(dim=-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z)[0].unsqueeze(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = attention_vectors.squeeze(dim=-1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(1)
        out = ((out * attention_vectors).sum(dim=-1)).squeeze(dim=-1)
        return out



# non-casual res-block

class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 9), causal=False, pad_mode='reflect'):
        super(ResBlock, self).__init__()
        self.convs1 = nn.Sequential(
            Snake1d(channels),
            SConv1d(chan_in=channels, chan_out=channels, kernel_size=kernel_size, dilation=dilation[0], causal=causal, pad_mode=pad_mode),
            Snake1d(channels),
            SConv1d(chan_in=channels, chan_out=channels, kernel_size=kernel_size, dilation=dilation[1], causal=causal, pad_mode=pad_mode),
            Snake1d(channels),
            SConv1d(chan_in=channels, chan_out=channels, kernel_size=kernel_size, dilation=dilation[2], causal=causal, pad_mode=pad_mode)
        )


    def forward(self, x):
        x = x + self.convs1(x)
        return x


# Selective Down-sampling Back-projection 
class SDBP_EncoderBlock(nn.Module):
    def __init__(self, chan_in, chan_out, stride, dilations=(1, 3, 9), causal=False, pad_mode='reflect'):
        super().__init__()
        self.downs_one = SConv1d(chan_in, chan_out, 2 * stride, causal=causal, pad_mode=pad_mode, stride=stride)
        self.downs_two = SConv1d(chan_in, chan_out, 2 * stride, causal=causal, pad_mode=pad_mode, stride=stride)
        self.ups_one = SConvTranspose1d(chan_out, chan_in, 2 * stride, stride=stride)
        self.res_one = ResBlock(chan_in, kernel_size=3, dilation=dilations, causal=causal, pad_mode=pad_mode)
        
        self.res_four = nn.Sequential(
            ResBlock(chan_out, kernel_size=3, dilation=dilations, causal=causal, pad_mode=pad_mode),
            Snake1d(chan_out)
        )
        self.skn = SelectNet(chan_out, kernel_size=3, causal=causal, pad_mode=pad_mode)

    def forward(self, x):
        x_downs_one = self.downs_one(x)
        x_ups_one = self.ups_one(x_downs_one)
        x_res = x - x_ups_one
        x_res_one = self.res_one(x_res)
       
        x_downs_two = self.downs_two(x_res_one)
        
        x_downs_three = x_downs_one + x_downs_two
        x_f = self.skn(x_downs_three, x_downs_one, x_downs_two)
        x_f = self.res_four(x_f)
        return x_f

#Selective Up-sampling Back-projection
class SUBP_DecoderBlock(nn.Module):
    def __init__(self, chan_in, chan_out, stride, dilations=(1, 3, 9), causal=False, pad_mode='reflect'):
        super().__init__()
        self.downs_one = SConv1d(chan_out, chan_in, 2 * stride, causal=causal, pad_mode=pad_mode, stride=stride)
        self.ups_one = SConvTranspose1d(chan_in, chan_out, 2 * stride, stride=stride)
        self.ups_two = SConvTranspose1d(chan_in, chan_out, 2 * stride, stride=stride)
        self.res_one = ResBlock(chan_in, kernel_size=3, dilation=dilations, causal=causal, pad_mode=pad_mode)
        
        self.res_four = nn.Sequential(
            ResBlock(chan_out, kernel_size=3, dilation=dilations, causal=causal, pad_mode=pad_mode),
            Snake1d(chan_out)
        )
        self.skn = SelectNet(chan_out, kernel_size=3, causal=causal, pad_mode=pad_mode)

    def forward(self, x):
        x_ups_one = self.ups_one(x)
        x_downs_one = self.downs_one(x_ups_one)
        x_res = x - x_downs_one
        x_res_one = self.res_one(x_res)
    
        x_ups_two = self.ups_two(x_res_one)
       
        x_ups_three = x_ups_one + x_ups_two
        x_f = self.skn(x_ups_three, x_ups_one, x_ups_two)
        x_f = self.res_four(x_f)
        return x_f
 


class Supercodec(nn.Module):
    def __init__(
            self,
            *,
            channels=32,
            strides=(2, 4, 5, 8),
            channel_mults=(2, 4, 8, 16),
            codebook_dim=128,
            codebook_size=1024,
            codebook_size_res=512,
            rq_num_quantizers=2,
            input_channels=1,
            enc_cycle_dilations=(1, 3, 9),
            dec_cycle_dilations=(1, 3, 9),
            target_sample_hz=16000,
            shared_codebook=False,
            training=False,
            causal=False,
            pad_mode='reflect',
    ):
        super().__init__()

        # for autosaving the config

        _locals = locals()
        _locals.pop('self', None)
        _locals.pop('__class__', None)
        self._configs = pickle.dumps(_locals)

        # rest of the class

        self.target_sample_hz = target_sample_hz  # for resampling on the fly

        self.single_channel = input_channels == 1
        self.strides = strides

        layer_channels = tuple(map(lambda t: t * channels, channel_mults))
        layer_channels = (channels, *layer_channels)
        chan_in_out_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        encoder_blocks = [SConv1d(input_channels, channels, 7, causal=causal, pad_mode=pad_mode)]

        for ((chan_in, chan_out), layer_stride) in zip(chan_in_out_pairs, strides):
            encoder_blocks.append(SDBP_EncoderBlock(chan_in, chan_out, layer_stride, enc_cycle_dilations, causal=causal, pad_mode=pad_mode))

        encoder_blocks += [SConv1d(layer_channels[-1], codebook_dim, 7, causal=causal, pad_mode=pad_mode)]

        self.encoder = nn.Sequential(*encoder_blocks)


       
        self.rq = ResidualVQ(
            dim=codebook_dim,
            num_quantizers=rq_num_quantizers,
            codebook_size=codebook_size,
            kmeans_init=True,
            kmeans_iters=10,
            shared_codebook = shared_codebook,
        )


        decoder_blocks = [SConv1d(codebook_dim, layer_channels[-1], 7, causal=causal, pad_mode=pad_mode)]
       

        for ((chan_in, chan_out), layer_stride) in zip(reversed(chan_in_out_pairs), reversed(strides)):
            decoder_blocks.append(SUBP_DecoderBlock(chan_out, chan_in, layer_stride, dec_cycle_dilations, causal=causal, pad_mode=pad_mode))
           
        decoder_blocks += [SConv1d(channels, input_channels, 7, causal=causal, pad_mode=pad_mode)]
        self.decoder = nn.Sequential(*decoder_blocks)

    

        # decoder
        self.decoder_blocks = decoder_blocks

        self.training = training
        self.codebook_dim = codebook_dim // 2 if split_res else codebook_dim

        self.apply(init_weights)

    @property
    def configs(self):
        return pickle.loads(self._configs)

    def decode_from_codebook_indices(self, quantized_indices):
        codes = self.rq.get_codes_from_indices(quantized_indices)
        x = reduce(codes, 'q ... -> ...', 'sum')
        x = rearrange(x, 'b n c -> b c n')
        return self.decoder(x)

    def save(self, path):
        path = Path(path)
        pkg = dict(
            model=self.state_dict(),
            config=self._configs,
            version=__version__
        )

        torch.save(pkg, str(path))


    def load_from_trainer_saved_obj(self, path):
        path = Path(path)
        assert path.exists()
        obj = torch.load(str(path))
        self.load_state_dict(obj['model'])

    def non_discr_parameters(self):
        return [
            *self.encoder.parameters(),
            *self.decoder.parameters()
        ]

    @property
    def seq_len_multiple_of(self):
        return functools.reduce(lambda x, y: x * y, self.strides)

    def forward(
            self,
            x,
            return_encoded=False,
            input_sample_hz=None,
    ):
        start_time = time.time()
        x, ps = pack([x], '* n')

        if exists(input_sample_hz):
            x = resample(x, input_sample_hz, self.target_sample_hz)

        x = curtail_to_multiple(x, self.seq_len_multiple_of)
        if x.ndim == 2:
            x = rearrange(x, 'b n -> b 1 n')
        x = self.encoder(x)


        x = rearrange(x, 'b c n -> b n c')
        
        x_new, indice, commit_loss = self.rq(x)
        indices = [indice]
        commit_losses = [commit_loss]
        if return_encoded:
            return x_new, indices
        
        x_new = rearrange(x_new, 'b c n -> b n c')
        
        recon_x = self.decoder(x_new)

        recon_x, = unpack(recon_x, ps, '* c n')
        if self.training:
            return recon_x, commit_losses
        return recon_x

if __name__ == "__main__":
    x = torch.randn(8, 16000)
    supercodec = Supercodec(strides=(4,4,4,5), channel_mults=(4,4,8,8))
    y = supercodec(x)
    print(y.shape)
    for n, m in supercodec.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f"{p/1e6:<.3f} M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(supercodec)
    print("Total # of params:", sum([np.prod(p.size()) for p in  supercodec.parameters()]))
