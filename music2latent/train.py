import datetime
import os
import shutil
import glob
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

from . import fad_utils
from . import misc
from .ema import ExponentialMovingAverage
from .hparams import hparams
from .utils import *
from .models import *
from .data import *
from .audio import *

if hparams.torch_compile_cache_dir is not None:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = hparams.torch_compile_cache_dir

torch.backends.cudnn.benchmark = True

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group


class Trainer:
    def __init__(self):
        if hparams.multi_gpu:
            torch.multiprocessing.set_start_method('spawn')
            misc.init()
        np.random.seed((hparams.seed * misc.get_world_size() + misc.get_rank()) % (1 << 31))
        torch.manual_seed(np.random.randint(1 << 31))

        # set self.device to cuda if it is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batch_size_per_gpu = hparams.batch_size // misc.get_world_size()

        self.save_path = None
        self.dl = get_dataloader(batch_size_per_gpu)
        if misc.get_rank()==0:
            self.ds_test = get_test_dataset()
        self.get_models()
        self.switch_save_checkpoint = True
        self.step = 0
        
        # INITIALIZE CHECKPOINT FOLDER
        if misc.get_rank()==0:
            if self.save_path is None:
                self.save_path = f'{hparams.checkpoint_path}/{str(datetime.datetime.now())}'
                os.makedirs(self.save_path, exist_ok=True)
                os.makedirs(os.path.join(self.save_path, 'code'), exist_ok=True)
                for file in glob.glob(os.path.dirname(__file__) + '/*.py'):
                    shutil.copyfile(file, self.save_path+'/code/'+os.path.basename(file))
            self.writer = SummaryWriter(log_dir=self.save_path)


    @torch.compile(mode='max-autotune-no-cudagraphs', disable=not hparams.compile_model)
    def forward_pass_consistency(self, data_encoder, noisy_samples, noisy_samples_plus_one, sigmas_step, sigmas):

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hparams.mixed_precision):

            if hparams.multi_gpu:
                fdata, fdata_plus_one = self.ddp(data_encoder, noisy_samples, noisy_samples_plus_one, sigmas_step, sigmas)
            else:
                fdata, fdata_plus_one = self.gen(data_encoder, noisy_samples, noisy_samples_plus_one, sigmas_step, sigmas)
            
            loss_weight = get_loss_weight(sigmas, sigmas_step)
            loss = huber(fdata,fdata_plus_one,loss_weight)
        return loss


    def train_it(self, wv):

        data = to_representation(wv)
        data_encoder = to_representation_encoder(wv)

        step = get_step_schedule(min(self.it,hparams.total_iters))
        self.step = step
        
        if hparams.use_lognormal:
            arbitrary_high_number = 10000
            w = get_sampling_weights(arbitrary_high_number, device=data.device)
            inds = torch.multinomial(w, data.shape[0], replacement=True).float()
            inds = (inds+torch.rand_like(inds))/float(arbitrary_high_number-1)
        else:
            inds = torch.rand((data.shape[0],), dtype=torch.float32, device=data.device)
            
        sigmas = get_sigma_continuous(inds)
        inds_step = get_step_continuous(inds, step)
        sigmas_step = get_sigma_continuous(inds_step)
        
        noises = torch.randn_like(data)
        noisy_samples = add_noise(data, noises, sigmas_step)
        noisy_samples_plus_one = add_noise(data, noises, sigmas)

        with misc.ddp_sync(self.ddp, ((self.it+1) % hparams.accumulate_gradients==0) or (self.it+1==len(self.dl))):
            loss = self.forward_pass_consistency(data_encoder, noisy_samples, noisy_samples_plus_one, sigmas_step, sigmas)
        self.scaler.scale(loss.float()).backward()
        loss = loss.detach().cpu().item()

        grad_norm = get_grad_norm(self.gen.parameters())
        if ((self.it+1) % hparams.accumulate_gradients==0) or (self.it+1==len(self.dl)):
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        
        if hparams.enable_ema and misc.get_rank()==0:
            self.ema.update()

        if misc.get_rank()==0:
            self.writer.add_scalar('loss', loss, self.it)
            self.writer.add_scalar('gradient norm', grad_norm.item(), self.it)
            self.writer.add_scalar('consistency step', step, self.it)

        return loss


    def train(self):

        self.gen.train()
        g = 0

        while self.it<hparams.total_iters:
            if hparams.multi_gpu:
                self.dl.sampler.set_epoch(self.epoch)
            loss_list = []

            if misc.get_rank()==0:
                pbar = tqdm(self.dl, desc=f'Iters {self.it}/{hparams.total_iters}', leave=True)
            else:
                pbar = self.dl

            for batchi,(x) in enumerate(pbar):

                self.update_learning_rate()
                
                loss = self.train_it(x.to(self.device))
                if misc.get_rank()==0:
                    self.writer.add_scalar('epoch', self.epoch, self.it)
                    self.writer.add_scalar('learning rate', self.optimizer.param_groups[0]['lr'], self.it)
                loss_list.append(loss)

                g += 1
                self.it += 1

                if batchi%100==0 and misc.get_rank()==0:
                    pbar.set_postfix({'loss': np.mean(loss_list[-g:], axis=0)})
            
            self.epoch = self.epoch + 1
            if misc.get_rank()==0:
                self.calculate_fad(hparams.inference_diffusion_steps)
                self.save_checkpoint(np.mean(loss_list[-g:]))
                if hparams.enable_ema:
                    with self.ema.average_parameters():
                        self.test_model()
                else:
                    self.test_model()

            g = 0

    def update_learning_rate(self):
        if self.it < hparams.warmup_steps:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = hparams.lr * (self.it / hparams.warmup_steps)
        else:
            if hparams.lr_decay == 'cosine':
                decay_iters = hparams.total_iters - hparams.warmup_steps
                current_iter = (self.it - hparams.warmup_steps) % (hparams.total_iters - hparams.warmup_steps)
                new_learning_rate = hparams.final_lr + (0.5 * (hparams.lr - hparams.final_lr) * (1. + np.cos((current_iter / decay_iters) * np.pi)))
            elif hparams.lr_decay == 'linear':
                decay_iters = hparams.total_iters - hparams.warmup_steps
                current_iter = (self.it - hparams.warmup_steps) % (hparams.total_iters - hparams.warmup_steps)
                new_learning_rate = hparams.lr - ((hparams.lr - hparams.final_lr) * (current_iter / decay_iters))
            elif hparams.lr_decay == 'inverse_sqrt':
                new_learning_rate = hparams.lr * (hparams.warmup_steps ** 0.5) / max(self.it, hparams.warmup_steps) ** 0.5
            elif hparams.lr_decay is None:
                new_learning_rate = hparams.lr
            else:
                raise ValueError('lr_decay must be None, "cosine", "linear", or "inverse_sqrt"')
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_learning_rate

    def test_model(self):
        self.gen.eval()
        max_steps = hparams.inference_diffusion_steps

        num_examples = 4
        original,reconstructed = encode_decode(self.gen, self.ds_test, num_examples)
        if len(original[0].shape)==2:
            original = [el[0,:] for el in original]
        if len(reconstructed[0].shape)==2:
            reconstructed = [el[0,:] for el in reconstructed]
        fig = plot_audio_compare(original,reconstructed)
        fig.suptitle(f'{max_steps} steps')
        if self.writer is not None:
            for ind in range(num_examples):
                self.writer.add_audio(f"original_{ind}", original[ind].detach().cpu().squeeze().numpy(), global_step=self.it, sample_rate=hparams.sample_rate)
                self.writer.add_audio(f"reconstructed_{ind}", reconstructed[ind].detach().cpu().squeeze().numpy(), global_step=self.it, sample_rate=hparams.sample_rate)
            self.writer.add_figure(f"figs/{max_steps}_steps", fig, global_step=self.it)
        plt.close()
        self.gen.train()

    def save_batch_to_wav(self, batch):
        print('Saving audio samples...')
        self.final_fad_path = os.path.join(self.save_path, hparams.eval_samples_path)
        os.makedirs(self.final_fad_path, exist_ok=True)
        for i in range(len(batch)):
            audio_data = batch[i].numpy()
            if len(audio_data.shape)==2:
                audio_data = audio_data[0]
            audio_data = (audio_data * 32767.0).astype(np.int16)  # Scale to 16-bit PCM range
            audio_file_path = os.path.join(self.final_fad_path, f'audio_{i}.wav')
            # Save the audio file
            write(audio_file_path, hparams.sample_rate, audio_data)
            
    def calculate_fad(self, diffusion_steps=1, log=True):
        if hparams.enable_ema:
            with self.ema.average_parameters():
                self.gen.eval()
                samples = encode_decode_batch(self.gen, self.ds_test, hparams.num_samples_fad, diffusion_steps=diffusion_steps)
                self.gen.train()
        else:
            self.gen.eval()
            samples = encode_decode_batch(self.gen, self.ds_test, hparams.num_samples_fad, diffusion_steps=diffusion_steps)
            self.gen.train()
        self.save_batch_to_wav(samples)
        score = fad_utils.compute_fad(self.final_fad_path)
        print(f'FAD: {score}')
        if log:
            for i in range(len(hparams.fad_models)):
                self.writer.add_scalar(f'fad_{hparams.fad_models[i]}', score[i], self.it)
        score = score[0]
        self.current_score = score
        if score<=self.score:
            self.switch_save_checkpoint = True
            self.score = score
            
    def save_checkpoint(self, loss):
        old_checkpoint_list = glob.glob(os.path.join(self.save_path, '*.pt'))

        os.makedirs(self.save_path, exist_ok=True)
        save_dict = {
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'loss': loss,
                    'fid': self.current_score,
                    'it': self.it,
                    'epoch': self.epoch,
                    'score': self.score,
                    'best_checkpoint_path' : self.best_checkpoint_path,
                    }
        save_dict['gen_state_dict'] = self.gen.state_dict()
        if hparams.enable_ema:
            save_dict['ema_state_dict'] = self.ema.state_dict()
        self.current_checkpoint_path = self.save_path+f'/model_fid_{self.current_score}_loss_{str(loss)[:6]}_iters_{self.it}.pt'
        torch.save(save_dict, self.current_checkpoint_path)
        del save_dict

        if self.switch_save_checkpoint:
            self.best_checkpoint_path = self.current_checkpoint_path
            self.switch_save_checkpoint = False

        if len(old_checkpoint_list)>0:
            for el in old_checkpoint_list:
                if os.path.basename(el)!=os.path.basename(self.best_checkpoint_path):
                    os.remove(el)
        
    def save_checkpoint_clean(self):
        os.makedirs(hparams.clean_path, exist_ok=True)
        save_dict = {
                    'fid': self.current_score,
                    'it': self.it,
                    'epoch': self.epoch,
                    }
        if hparams.enable_ema:
            self.ema.copy_to()
            with self.ema.average_parameters():
                save_dict['gen_state_dict'] = self.gen.state_dict()
        final_path = hparams.clean_path+f'/clean_model_fid_{self.current_score}_iters_{self.it}.pt'
        torch.save(save_dict, final_path)

    @torch.no_grad()
    def load_matching_weights(self, model, state_dict):
        """
        Load weights from state_dict into model, only when module names match and parameter sizes match.
        Leave model parameters untouched if no match is found.
        """
        model_dict = model.state_dict()
        loaded_count = 0
        total_count = len(model_dict.keys())
        for name, param in state_dict.items():
            if name in model_dict:
                if param.shape == model_dict[name].shape:
                    model_dict[name].copy_(param)
                    loaded_count += 1
        model.load_state_dict(model_dict)
        print(f"Loaded {loaded_count} out of {total_count} layers from state_dict.")

    def get_models(self):
        gen = UNet().to(self.device)
        gen.train().requires_grad_(True)
        if hparams.multi_gpu:
            self.ddp = DDP(gen, device_ids=[self.device], broadcast_buffers=False, find_unused_parameters=True)
        else:
            self.ddp = None
        optimizer = torch.optim.RAdam(gen.parameters(), lr=hparams.lr, betas=(hparams.optimizer_beta1, hparams.optimizer_beta2))
        scaler = torch.amp.GradScaler(enabled=hparams.mixed_precision)
        ema = ExponentialMovingAverage(gen.parameters(), decay=hparams.ema_momentum, use_num_updates=hparams.warmup_ema)
        it = 0
        epoch = 0
        score = 1e7
        best_checkpoint_path = ''
        
        if hparams.load_path is not None:
            checkpoint = torch.load(hparams.load_path, map_location=torch.device('cpu'))
            self.load_matching_weights(gen, checkpoint['gen_state_dict'])
            if hparams.load_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if hparams.load_ema:
                ema.load_state_dict(checkpoint['ema_state_dict'])
            if hparams.load_iter:
                self.save_path = os.path.dirname(hparams.load_path)
                it = checkpoint['it']
                epoch = checkpoint['epoch']
                score = checkpoint['score']
                best_checkpoint_path = checkpoint['best_checkpoint_path']
            del checkpoint

        if misc.get_rank()==0:
            total_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
            print(f"Total number of trainable parameters in Generator: {total_params}")
        
        self.optimizer = optimizer
        self.scaler = scaler
        self.ema = ema
        self.it = it
        self.epoch = epoch
        self.score = score
        self.current_score = score
        self.best_checkpoint_path = best_checkpoint_path
        self.gen = gen



def main():
    trainer = Trainer()
    trainer.train()
    if hparams.multi_gpu:
        destroy_process_group()

def main_fad():
    trainer = Trainer()
    trainer.calculate_fad()

def main_save_clean():
    trainer = Trainer()
    trainer.save_checkpoint_clean()