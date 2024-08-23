import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
from scheduler import WarmupCosineLrScheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from utils import AttrDict, build_env
from utils import  get_dataset_filelist, mel_spectrogram
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

from supercodec import Supercodec
from data import SoundDataset, get_dataloader
from msstftd import MultiScaleSTFTDiscriminator
from losses import total_loss, disc_loss

torch.backends.cudnn.benchmark = True

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'], world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))


    supercodec = Supercodec(
        codebook_size=h.codebook_size,
        codebook_dim=h.codebook_dim,
        rq_num_quantizers=h.rq_num_quantizers,
        shared_codebook = False,
        strides=h.strides,
        channel_mults=h.channel_mults,
        training=True
        )

    supercodec = supercodec.to(device)

    # DDP(model, device_ids=[rank], find_unused_parameters=True)

    disc_model = MultiScaleSTFTDiscriminator(filters=h.filters)
    disc_model = disc_model.to(device)
    
    if rank == 0:
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        supercodec.load_state_dict(state_dict_g['generator'])
        disc_model.load_state_dict(state_dict_do['discriminator'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        supercodec = DistributedDataParallel(supercodec, device_ids=[rank], find_unused_parameters=True).to(device)
        disc_model = DistributedDataParallel(disc_model, device_ids=[rank], find_unused_parameters=True).to(device)
        
    params = [p for p in supercodec.parameters() if p.requires_grad]
    disc_params = [p for p in disc_model.parameters() if p.requires_grad]
    # optim_g = torch.optim.AdamW([supercodec.encoder.parameters(), supercodec.decoder.parameters()], h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_g = torch.optim.Adam([{'params': params, 'lr': h.lr}], betas=(0.5, 0.9))

    optim_d = torch.optim.Adam([{'params': disc_params, 'lr': h.disc_lr}], betas=(0.5, 0.9))

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    training_filelist, validation_filelist = get_dataset_filelist(a)


    trainset = SoundDataset(
            training_filelist,
            split=True,
            shuffle=True,
            segment_size=h.segment_size,
            max_length=a.data_max_length
        )

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:

        validset = SoundDataset(
            validation_filelist,
            split=False,
            shuffle=True,
            segment_size=h.segment_size,
            max_length=a.data_max_length
        )
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

    sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
    
    apply_grad_penalty = a.apply_grad_penalty_every > 0 and not (steps % a.apply_grad_penalty_every)

    scheduler_g = WarmupCosineLrScheduler(optim_g, max_iter=h.max_epoch * len(train_loader), eta_ratio=0.1,
                                        warmup_iter=h.warmup_epoch * len(train_loader),
                                        warmup_ratio=1e-4)
    scheduler_d = WarmupCosineLrScheduler(optim_d, max_iter=h.max_epoch * len(train_loader),
                                             eta_ratio=0.1,
                                             warmup_iter=h.warmup_epoch * len(train_loader),
                                             warmup_ratio=1e-4)

    supercodec.train()
    disc_model.train()
    logs = {}
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()

            wave = batch
            wave = torch.autograd.Variable(wave.to(device, non_blocking=True))


            recon_g, loss_w = supercodec(wave, return_recons_only=True)

            wave = wave.unsqueeze(1)

            #Discriminator
            logits_real, fmap_real = disc_model(wave)
            optim_d.zero_grad()
            logits_fake, fmap_fake = disc_model(recon_g.detach())
            loss_disc = disc_loss([logit_real for logit_real in logits_real], logits_fake)
            loss_disc.backward(retain_graph=True)
            optim_d.step()

            
            # Generator
            optim_g.zero_grad()
            logits_fake, fmap_fake = disc_model(recon_g)
            loss_g = total_loss(fmap_real, logits_fake, fmap_fake, wave, recon_g)
            
            wave_mel = mel_spectrogram(wave.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)

            recon_g_mel = mel_spectrogram(recon_g.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)

            loss_mel = F.l1_loss(wave_mel, recon_g_mel) * 45
            loss = loss_g + loss_w
            loss.backward()
            optim_g.step()

            scheduler_g.step()
            scheduler_d.step()


            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(wave_mel,recon_g_mel).item()

                    
                    print('Steps : {:d}, Gen Loss: {:.3f}, VQ. Loss : {:.3f},  Mel-Spec. Error : {:4.3f}, Disc. Error : {:4.3f}, s/b : {:4.3f}'.
                                format(steps, loss_g.item(), loss_w.item(),
                                    mel_error, loss_disc.item(), time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (supercodec.module if h.num_gpus > 1 else supercodec).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {
                                    'discriminator': (
                                            disc_model.module if h.num_gpus > 1 else disc_model).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    sw.add_scalar("training/disc_loss", loss_disc, steps)
                    sw.add_scalar("training/loss_g", loss_g, steps)
                    sw.add_scalar("training/all_commit_loss", loss_w, steps)


                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    supercodec.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            wave = batch
                            wave = wave.to(device)
                            recons = supercodec(wave, return_recons_only = True)
                            wave_mel = mel_spectrogram(wave.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size,
                                                          h.fmin, h.fmax_for_loss)
                            y_g_hat_mel = mel_spectrogram(recons.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size,
                                                          h.fmin, h.fmax_for_loss)

                            val_err_tot += F.l1_loss(wave_mel, y_g_hat_mel).item()
                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), wave[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(wave_mel.squeeze(0).cpu().numpy()), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), recons[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(recons.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)


                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)
                    supercodec.train()
            steps += 1
        

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default="")
    parser.add_argument('--input_wavs_dir_validation', default="")
    parser.add_argument('--input_training_file', default="")
    parser.add_argument('--input_validation_file',
                        default="")
    parser.add_argument('--checkpoint_path', default='')
    parser.add_argument('--config', default='config_v1.json')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--apply_grad_penalty_every', default=4, type=int)
    parser.add_argument('--data_max_length', default=32000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=True, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
