import numpy as np
import sys
import os, cv2, csv
from DAVE2pytorch import DAVE2PytorchModel
import kornia

from PIL import Image
import copy
from scipy import stats
import torch.utils.data as data
from pathlib import Path
import skimage.io as sio
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import random
from torch import Tensor
from torchvision.transforms import Compose, ToTensor, PILToTensor, functional as transforms
from io import BytesIO
import skimage


def stripleftchars(s):
    for i in range(len(s)):
        if s[i].isnumeric():
            return s[i:]
    return -1


def striplastchars(s):
    s = s.split("-")[-1]
    for i in range(len(s)):
        if s[i].isnumeric():
            return s[i:]
    return -1


class MultiDirectoryDataSequence(data.Dataset):
    def __init__(self, root=None, RRL_dir=None, image_size=(100,100), transform=None,
                 robustification=False, noise_level=10, sample_id="PREDICTION", effect=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sample_id = sample_id
        self.root = root
        self.RRL_dir = RRL_dir
        self.effect = effect
        self.transform = transform
        self.robustification = robustification
        self.noise_level = noise_level
        self.size = 0
        self.image_size = image_size
        self.image_paths_hashmap = {}
        self.all_image_paths = []
        self.dfs_hashmap = {}
        self.dirs = []
        if self.root is not None:
            self.process_basemodel_dirs()
        if RRL_dir is not None:
            self.process_RRL_dir()
        self.cache = {}
        self.min_transform = Compose([ToTensor()])


    def process_basemodel_dirs(self, marker=""):
        for p in Path(self.root).iterdir():
            if p.is_dir() and marker in str(p):
                self.dirs.append("{}/{}".format(p.parent, p.stem.replace(marker, "")))
                image_paths = []
                try:
                    self.dfs_hashmap[f"{p}"] = pd.read_csv(f"{p}/data.csv")
                except FileNotFoundError as e:
                    try:
                        self.dfs_hashmap[f"{p}"] = pd.read_csv(f"{p}/data.txt")
                    except FileNotFoundError as e:
                        print(e, "\nNo data.csv or data.txt in directory")
                        continue
                for pp in Path(p).iterdir():
                    if pp.suffix.lower() in [".jpg", ".png", ".jpeg",
                                             ".bmp"] and "base" in pp.name:
                        image_paths.append(pp)
                        self.all_image_paths.append(pp)
                image_paths.sort(key=lambda p: int(stripleftchars(p.stem)))
                self.image_paths_hashmap[p] = copy.deepcopy(image_paths)
                self.size += len(image_paths)


    def process_RRL_dir(self):
        skipcount = 0
        if self.RRL_dir is not None and Path(self.RRL_dir).is_dir():
            for p in Path(self.RRL_dir).iterdir():
                if p.is_dir() and "tb_logs_DDPG" not in str(p):
                    # ep = int(str(p.stem).replace("ep", ""))
                    # if ep % 8 != 0:
                    self.dirs.append("{}/{}".format(p.parent, p.stem))
                    image_paths = []
                    try:
                        self.dfs_hashmap[f"{p}"] = pd.read_csv(f"{p}/data.csv")
                    except FileNotFoundError as e:
                        try:
                            self.dfs_hashmap[f"{p}"] = pd.read_csv(f"{p}/data.txt")
                        except FileNotFoundError as e:
                            print(e, "\nNo data.csv or data.txt in directory")
                            continue
                    for pp in Path(p).iterdir():
                        if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"] and "sample-base" in pp.name:
                            image_paths.append(pp)
                            self.all_image_paths.append(pp)
                    image_paths.sort(key=lambda p: int(striplastchars(p.stem)))
                    self.image_paths_hashmap[p] = copy.deepcopy(image_paths)
                    self.size += len(image_paths)

    def get_total_samples(self):
        return self.size

    def get_directories(self):
        return self.dirs
        
    def __len__(self):
        return len(self.all_image_paths)
    
    def robustify(self, image_base, y_steer):
        if random.random() > 0.75:
            image_base = torch.flip(image_base, (2,))
            y_steer = -y_steer
        if random.random() > 0.75:
            gauss = kornia.filters.GaussianBlur2d((3, 3), (1.5, 1.5))
            image_base = gauss(image_base[None])[0]
        if random.random() > 0.75:
            gauss = kornia.filters.GaussianBlur2d((3, 3), (1.5, 1.5))
        if self.noise_level is not None:
            image_base = torch.clamp(image_base + (torch.randn(*image_base.shape) / self.noise_level), 0, 1)
        return image_base, y_steer

    def reverse(self):
        self.all_image_paths.reverse()

    def __getitem__(self, idx):
        if idx in self.cache:
            if self.robustification:
                sample = self.cache[idx]
                y_steer = sample["steering_input"]
                image_base = copy.deepcopy(sample["image_base"])
                image_base_rb, image_transf_rb, y_steer_rb = self.robustify(image_base, y_steer)
                return {"image_base": image_base_rb, "steering_input": y_steer_rb, "throttle_input": sample["throttle_input"]} #,  "img_name": sample["img_name"], "all": torch.FloatTensor([y_steer, sample["throttle_input"]])}
            else:
                return self.cache[idx]
        img_name = self.all_image_paths[idx]
        image = Image.open(img_name)
        image_base = image.resize(self.image_size)
        image_base = self.transform(image_base)
        pathobj = Path(img_name)
        df = self.dfs_hashmap[f"{pathobj.parent}"]
        df_index = df.index[df['IMG'] == img_name.name]
        # print(f"{img_name}")
        # print(f"{img_name=} \t {idx} \t {df_index=} \t {df.loc[df_index, self.sample_id]=}")
        # print(f"{df.columns=}")
        # df.columns=Index(['IMG', 'PREDICTION', 'STEERING_INPUT', 'POSITION', 'ORIENTATION', 'KPH',
        #        'STEERING_ANGLE_CURRENT', 'THROTTLE_INPUT'], dtype='object')
        try:
            orig_y_steer = df.loc[df_index, self.sample_id].item()
            y_throttle = df.loc[df_index, 'THROTTLE_INPUT'].item()
            y_steer = copy.deepcopy(orig_y_steer)
            # print(f"INDEX {df_index}\n\t{df.loc[df_index, 'PREDICTION'].item()=} \n\t{df.loc[df_index, 'STEERING_INPUT'].item()=} \n\t{df.loc[df_index, 'STEERING_ANGLE_CURRENT'].item()=}")

            img_name_raw = Tensor(np.array([ord(c) for c in str(img_name)]))
            # print(f"{img_name_raw.shape=}")
            filename_char_length = 750
            img_name_tensor = torch.zeros((1,filename_char_length))
            img_name_raw = torch.nn.functional.pad(input=img_name_raw[None], pad=(0, filename_char_length - img_name_raw.shape[0], 0, 0), mode='constant', value=0)
            img_name_tensor = img_name_tensor + img_name_raw
            if self.robustification:
                image_base_rb, y_steer_rb = self.robustify(copy.deepcopy(image_base), y_steer)
                sample = {"image_base": image_base_rb, "steering_input": y_steer_rb, "throttle_input": y_throttle}
            else:
                sample = {"image_base": image_base, "steering_input": orig_y_steer, "throttle_input": y_throttle, "img_name": img_name_tensor}

            orig_sample = {"image_base": image_base,  "steering_input": orig_y_steer, "throttle_input": y_throttle, "img_name": img_name_tensor}
            
            if sys.getsizeof(self.cache) < 8 * 1.0e10:
                self.cache[idx] = orig_sample
            else:
                print(f"{len(self.cache.keys())=}")
            return sample
        except Exception as e:
            print(e)
            print(f"{img_name=}")
            print(f"{self.sample_id=}")
            print(f"{len(self.cache.keys())=}")

    def get_outputs_distribution(self):
        all_outputs = np.array([])
        for key in self.dfs_hashmap.keys():
            df = self.dfs_hashmap[key]
            arr = df[self.sample_id].to_numpy()
            all_outputs = np.concatenate((all_outputs, arr), axis=0)
        all_outputs = np.array(all_outputs)
        moments = self.get_distribution_moments(all_outputs)
        return moments

    def remove_sample(self, idx):
        self.all_image_paths = [ip for ip in self.all_image_paths if not ip == idx]
        print("Removed", idx)

    ##################################################
    # ANALYSIS METHODS
    ##################################################

    # Moments are 1=mean 2=variance 3=skewness, 4=kurtosis
    def get_distribution_moments(self, arr):
        arr = list(arr)
        moments = {}
        moments['shape'] = np.asarray(arr).shape
        moments['mean'] = np.mean(arr)
        moments['median'] = np.median(arr)
        moments['var'] = np.var(arr)
        moments['skew'] = stats.skew(arr)
        moments['kurtosis'] = stats.kurtosis(arr)
        moments['max'] = max(arr)
        moments['min'] = min(arr)
        return moments
