import torch
from torch.utils.data import Dataset, DataLoader
import os
import soundfile as sf
import random
from tqdm import tqdm

from .hparams import hparams

from torch.utils.data.distributed import DistributedSampler

class TestAudioDataset(Dataset):
    def __init__(self, wav_path, hop, fac, data_length, tot_samples=None, random_sampling=True):
        self.random_sampling = random_sampling
        self.paths = find_files_with_extensions(wav_path, extensions=['.wav', '.flac'])
        # sort paths
        self.paths = sorted(self.paths)
        seed_value = 42
        shuffling_random = random.Random(seed_value)
        shuffling_random.shuffle(self.paths)
        self.data_samples = len(self.paths)
        print(f'Found {self.data_samples} samples.')
        self.hop = hop
        if tot_samples is None:
            self.tot_samples = self.data_samples
        else:
            self.tot_samples = tot_samples
        self.num_repetitions = self.tot_samples//self.data_samples
        self.wv_length = hop * data_length + (fac-1)*hop

    def __len__(self):
        return int(self.tot_samples)

    def __getitem__(self, idx):
        if idx>(self.data_samples*self.num_repetitions):
            idx = torch.randint(self.data_samples, size=(1,)).item()
        else:
            idx = idx%self.data_samples
        path = self.paths[idx]
        try:
            wv,_ = sf.read(path, dtype='float32', always_2d=True)
            if wv.shape[0]<self.wv_length:
                idx = torch.randint(self.tot_samples, size=(1,)).item()
                return self.__getitem__(idx)
            wv = torch.from_numpy(wv)
            # convert to mono
            if wv.shape[-1]>1:
                wv = wv.mean(dim=-1, keepdim=True)
            if wv.shape[-1]==1:
                wv = torch.cat([wv,wv], dim=1)
            wv = wv[:,:2]
            wv = wv.permute(1,0)
            # if not stereo:
            wv = wv[torch.randint(wv.shape[0], size=(1,)).item(),:]
        except Exception as e:
            print(e)
            idx = torch.randint(self.tot_samples, size=(1,)).item()
            return self.__getitem__(idx)
        return wv
    

def find_files_with_extensions(path, extensions=['.wav', '.flac']):
    found_files = []
    # Recursively traverse the directory
    for foldername, subfolders, filenames in tqdm(os.walk(path)):
        for filename in filenames:
            # Check if the file has an extension from the specified list
            if any(filename.lower().endswith(ext.lower()) for ext in extensions):
                # Build the full path to the file
                file_path = os.path.join(foldername, filename)
                found_files.append(file_path)

    return found_files


class AudioDataset(Dataset):
    def __init__(self, wav_paths, hop, fac, data_length, data_fractions, rms_min=0.001, data_extensions=['.wav', '.flac'], tot_samples=None, random_sampling=True):
        self.random_sampling = random_sampling
        if data_fractions is None:
            data_fractions = [1/len(wav_paths) for _ in wav_paths]
        tot_fractions = sum(data_fractions)
        data_fractions = [el/tot_fractions for el in data_fractions]
        self.tot_samples = tot_samples
        self.rms_min = rms_min
        self.paths = []
        self.num_samples = []
        self.num_tot_samples = []
        self.num_repetitions = []
        for path,fraction in zip(wav_paths,data_fractions):
            paths = find_files_with_extensions(path, extensions=data_extensions)
            seed_value = 42
            shuffling_random = random.Random(seed_value)
            shuffling_random.shuffle(paths)
            num_samples = len(paths)
            print(f'Found {num_samples} samples.')
            self.paths.append(paths)
            self.num_samples.append(num_samples)
            if tot_samples is None:
                self.num_tot_samples.append(int(num_samples))
            else:
                self.num_tot_samples.append(int(tot_samples))
            self.num_repetitions.append(self.num_tot_samples[-1]//num_samples)

        self.hop = hop
        self.data_length = data_length
        self.wv_length = hop * data_length + (fac-1)*hop
        self.data_fractions = torch.tensor(data_fractions)

    def __len__(self):
        return int(self.tot_samples)

    def __getitem__(self, idx):
        data_id = torch.multinomial(self.data_fractions, 1).item()
        if idx>(self.num_samples[data_id]*self.num_repetitions[data_id]):
            idx = torch.randint(self.num_samples[data_id], size=(1,)).item()
        else:
            idx = idx%self.num_samples[data_id]
        path = self.paths[data_id][idx]
        try:
            info = sf.info(path)
            samplerate = info.samplerate
            duration = info.duration
            length = int(samplerate*duration)
            rand_start = torch.randint(length-self.wv_length, size=(1,)).item()
            wv,_ = sf.read(path, frames=self.wv_length, start=rand_start, stop=None, dtype='float32', always_2d=True)
            wv = torch.from_numpy(wv)
            if wv.shape[-1]==1:
                wv = torch.cat([wv,wv], dim=1)
            wv = wv[:,:2]
            wv = wv.permute(1,0)

            # if not stereo:
            wv = wv[torch.randint(wv.shape[0], size=(1,)).item(),:]

            rms = torch.sqrt(torch.mean(wv**2))
            if rms < self.rms_min:
                idx = torch.randint(self.tot_samples, size=(1,)).item()
                return self.__getitem__(idx)

        except Exception as e:
            print(e)
            idx = torch.randint(self.tot_samples, size=(1,)).item()
            return self.__getitem__(idx)
        return wv


def get_dataloader(batch_size_per_gpu):
    
    dataset = AudioDataset(hparams.data_paths, hparams.hop, 4, hparams.data_length, hparams.data_fractions, hparams.rms_min, hparams.data_extensions, hparams.iters_per_epoch*hparams.batch_size)
    
    if hparams.multi_gpu:
        return DataLoader(dataset, batch_size=batch_size_per_gpu, drop_last=True, shuffle=False, sampler=DistributedSampler(dataset), num_workers=hparams.num_workers, pin_memory=True)
    else:
        return DataLoader(dataset, batch_size=batch_size_per_gpu, drop_last=True, shuffle=True, num_workers=hparams.num_workers, pin_memory=True)


def get_test_dataset():
    return TestAudioDataset(hparams.data_path_test, hparams.hop, 4, hparams.data_length)