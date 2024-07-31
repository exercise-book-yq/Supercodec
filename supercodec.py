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
from einops import rearrange, reduce, pack, unpack

from vector_quantize_pytorch import ResidualVQ
# from residual_vq import ResidualVQ

from utils import curtail_to_multiple
from utils import init_weights
from version import __version__
from packaging import version

parsed_version = version.parse(__version__)

import pickle


# helper functions
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


# Selective feature fusion
# Youqiang Zheng
# 2024.04.04
class SelectNet(nn.Module):
    def __init__(self, in_channels, kernel_size=3, M=2, r=2, stride=1, L=32, G=1):
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
            nn.Conv1d(in_channels, in_channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)),
            nn.ReLU(inplace=False)
            )
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv1d(in_channels, in_channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
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

class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                                           padding=get_padding(kernel_size, dilation[0]))), weight_norm(
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                      padding=get_padding(kernel_size, dilation[1]))), weight_norm(
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                      padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)

        # self.convs2 = nn.ModuleList(
        #     [weight_norm(
        #         nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
        #      weight_norm(
        #          nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
        #      weight_norm(
        #          nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])
        # self.convs2.apply(init_weights)

    def forward(self, x):
        for c1 in self.convs1:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                                          padding=get_padding(kernel_size, dilation[0]))), weight_norm(
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                      padding=get_padding(kernel_size, dilation[1])))])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)







# sound stream

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# causalconv1d
    
class CausalConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, **kwargs):
        super().__init__()
        kernel_size = kernel_size
        dilation = kwargs.get('dilation', 1)
        stride = kwargs.get('stride', 1)
        self.causal_padding = dilation * (kernel_size - 1) + 1 - stride
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, **kwargs)
        self.chan_in = chan_in
        self.chan_out = chan_out

    def forward(self, x):
        x = F.pad(x, (self.causal_padding, 0), mode='reflect')
        if self.chan_in != self.chan_out:
            x = self.conv(x)
            return x
        return self.conv(x)

#causal convtranspose1d

class CausalConvTranspose1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride, **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.padding = kernel_size - 1
        self.conv = nn.ConvTranspose1d(chan_in, chan_out, kernel_size, stride, **kwargs)

    def forward(self, x):
        n = x.shape[-1]

        out = self.conv(x)
        out = out[..., :(n * self.upsample_factor)]

        return out

#causal resblock

def ResidualUnit(chan_in, chan_out, dilation, kernel_size=7):
    return Residual(nn.Sequential(
        CausalConv1d(chan_in, chan_out, kernel_size, dilation=dilation),
        nn.ELU(),
        CausalConv1d(chan_out, chan_out, 1),
        nn.ELU()
    ))

#Encoderblock three residual blocks one causalconv1d
def EncoderBlock(chan_in, chan_out, stride, cycle_dilations=(1, 3, 9)):
    it = cycle(cycle_dilations)
    return nn.Sequential(
        ResidualUnit(chan_in, chan_in, next(it)),
        ResidualUnit(chan_in, chan_in, next(it)),
        ResidualUnit(chan_in, chan_in, next(it)),
        CausalConv1d(chan_in, chan_out, 2 * stride, stride=stride)
    )

#Decoderblock one causalconvtransposed three residual blocks
def DecoderBlock(chan_in, chan_out, stride, cycle_dilations=(1, 3, 9)):
    even_stride = (stride % 2 == 0)
    padding = (stride + (0 if even_stride else 1)) // 2
    output_padding = 0 if even_stride else 1

    it = cycle(cycle_dilations)
    return nn.Sequential(
        CausalConvTranspose1d(chan_in, chan_out, 2 * stride, stride=stride),
        ResidualUnit(chan_out, chan_out, next(it)),
        ResidualUnit(chan_out, chan_out, next(it)),
        ResidualUnit(chan_out, chan_out, next(it)),
    )


# Selective Down-sampling Back-projection 
class SBMP_Encoder(nn.Module):
    def __init__(self, chan_in, chan_out, stride, cycle_dilations=(1, 3, 5)):
        super().__init__()
        self.it = cycle(cycle_dilations)
        self.downs_one = CausalConv1d(chan_in, chan_out, 2 * stride, stride=stride)
        self.downs_two = CausalConv1d(chan_in, chan_out, 2 * stride, stride=stride)
        self.ups_one = CausalConvTranspose1d(chan_out, chan_in, 2 * stride, stride=stride)
        self.res_one = ResBlock2(chan_in)
        
        self.res_four = ResBlock1(chan_out)
        self.skn = SelectNet(chan_out)

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
class SBMP_Decoder(nn.Module):
    def __init__(self, chan_in, chan_out, stride, cycle_dilations=(1, 3, 5)):
        super().__init__()
        self.it = cycle(cycle_dilations)
        self.downs_one = CausalConv1d(chan_out, chan_in, 2 * stride, stride=stride)
        self.ups_one = CausalConvTranspose1d(chan_in, chan_out, 2 * stride, stride=stride)
        self.ups_two = CausalConvTranspose1d(chan_in, chan_out, 2 * stride, stride=stride)
        self.res_one = ResBlock2(chan_in)
        
        self.res_four = ResBlock1(chan_out)
        self.skn = SelectNet(chan_out)

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
            codebook_dim=512,
            codebook_size=1024,
            rq_num_quantizers=8,
            input_channels=1,
            enc_cycle_dilations=(1, 3, 9),
            dec_cycle_dilations=(1, 3, 9),
            target_sample_hz=16000,
            shared_codebook=False,
            training=False
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

        encoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(chan_in_out_pairs, strides):
            encoder_blocks.append(SBMP_Encoder(chan_in, chan_out, layer_stride, enc_cycle_dilations))
        self.encoder = nn.Sequential(
            CausalConv1d(input_channels, channels, 7),
            *encoder_blocks,
            CausalConv1d(layer_channels[-1], codebook_dim, 3)
        )


        
        self.rq = ResidualVQ(
            dim=codebook_dim,
            num_quantizers=rq_num_quantizers,
            codebook_size=codebook_size,
            kmeans_init=True,
            kmeans_iters=10,
            shared_codebook = shared_codebook,
        )


        decoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(reversed(chan_in_out_pairs), reversed(strides)):
            decoder_blocks.append(SBMP_Decoder(chan_out, chan_in, layer_stride, dec_cycle_dilations))


        self.decoder = nn.Sequential(
            CausalConv1d(codebook_dim, layer_channels[-1], 7),
            *decoder_blocks,
            CausalConv1d(channels, input_channels, 7)
        )

        # decoder
        self.decoder_blocks = decoder_blocks

        self.training = training

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
            return_recons_only=False,
            input_sample_hz=None,
            apply_grad_penalty=True
    ):
        start_time = time.time()
        x, ps = pack([x], '* n')

        if exists(input_sample_hz):
            x = resample(x, input_sample_hz, self.target_sample_hz)

        x = curtail_to_multiple(x, self.seq_len_multiple_of)
        if x.ndim == 2:
            x = rearrange(x, 'b n -> b 1 n')
        orig_x = x.clone()
        x = self.encoder(x)

        x = rearrange(x, 'b c n -> b n c')
        
        x_new, indices, commit_loss = self.rq(x)
        if return_encoded:
            return x_new, indices
        
        x_new = rearrange(x_new, 'b c n -> b n c')
        
        recon_x = self.decoder(x_new)
        
        if return_recons_only:
            recon_x, = unpack(recon_x, ps, '* c n')
            if self.training:
                return recon_x, commit_loss.sum()
            return recon_x
