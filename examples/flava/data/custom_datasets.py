import torch
import random
import os

from PIL import Image
from torch.utils.data import Dataset


class YFCCDataset(Dataset):
    def __init__(self, df, image_root, image_transform=None, text_transform=None, itm_probability=0.1,  ext='.jpg'):
        self.df = df
        self.image_root = image_root
        self.ext = ext
        self.itm_probability = itm_probability
        self.image_transform = image_transform
        self.text_transform = text_transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        torch.cuda.nvtx.range_push("YFCC-GetItem")
        image_key = self.df.iloc[idx, 0]
        image_filename = os.path.join(self.image_root, image_key[:3], image_key[3:6], image_key) + self.ext
        image = Image.open(image_filename)
        text = self.df.iloc[idx, 1]
        # Transforms
        output = {}
        if self.itm_probability > 0:
            output["itm_labels"] = torch.ones(1, dtype=torch.long)
        if random.random() < self.itm_probability:
            random_idx = random.randint(0, len(self.df) - 1)
            while idx == random_idx:
                text = self.df.iloc[random_idx, 1]
            output["itm_labels"] = torch.zeros(1, dtype=torch.long)

        output.update(self.image_transform(image))
        # TODO Need to refactor the logic here
        output["itm_labels"] = output["itm_labels"].squeeze()
        text_infos = self.text_transform(text)
        for info in text_infos:
            text_infos[info] = text_infos[info].squeeze()
        output.update(text_infos)
        torch.cuda.nvtx.range_pop()
        return output
