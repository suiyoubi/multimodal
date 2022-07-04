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
        image_key = self.df.iloc[idx, 0]
        image_filename = os.path.join(self.image_root, image_key[:3], image_key[3:6], image_key) + self.ext
        image = Image.open(image_filename)
        # Use description field if exists. Otherwise use title field instead.
        text = self.df.iloc[idx, 2] if self.df.iloc[idx, 2] is not None else self.df.iloc[idx, 1]
        # Transforms
        output = {}
        if self.itm_probability > 0:
            output["itm_labels"] = torch.ones(1, dtype=torch.long)
        if random.random() < self.itm_probability:
            random_idx = random.randint(0, len(self.df) - 1)
            while idx == random_idx:
                random_idx = random.randint(0, len(self.df) - 1)
                text = self.df.iloc[random_idx, 1]
            output["itm_labels"] = torch.zeros(1, dtype=torch.long)

        output.update(self.image_transform(image))
        # TODO Need to refactor the logic here
        output["itm_labels"] = output["itm_labels"].squeeze()
        try:
            text_infos = self.text_transform(text)
        except ValueError as e:
            text_infos = self.text_transform("")
            print(e)
            print(f"ERROR: text is {text} for idx {idx}, image filename is {image_filename}")
        for info in text_infos:
            text_infos[info] = text_infos[info].squeeze()
        output.update(text_infos)
        return output

if __name__ == '__main__':
    from .transforms import (
    default_text_transform,
    TEXT_DEFAULT_TOKENIZER,
    VL_MAX_LENGTH_DEFAULT,
    )
    from transformers import BertTokenizerFast
    text_tokenizer = BertTokenizerFast.from_pretrained(TEXT_DEFAULT_TOKENIZER) # should use BertTokenizerFast
    text_transform = default_text_transform(text_tokenizer, max_text_length=VL_MAX_LENGTH_DEFAULT)
    text = 'This+is+a+sample+sentence.'
    print(f'+ separator: {text_transform(text)}')
    text = 'This is a sample sentence.'
    print(f'space separator: {text_transform(text)}')