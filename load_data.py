import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
            the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
            T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.bos_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")

        # always have to load the nl data
        self.src_lines  = load_lines(os.path.join(data_folder, (split + ".nl")))

        if split != "test":
            self.tgt_lines  = load_lines(os.path.join(data_folder, (split + ".sql")))
        else:
            self.tgt_lines = None

        self.process_data(data_folder, split, self.tokenizer)


    def process_data(self, data_folder, split, tokenizer):
        align_path = os.path.join(data_folder, "alignment.txt")

    # Step 1: build replacement rules
        replacements = []
        if os.path.exists(align_path):
            with open(align_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = re.split(r"\t+", line)
                    if len(parts) >= 2:
                        src = parts[0].strip()
                        dst = parts[1].strip()
                        pat = re.compile(rf"(?<!\w){re.escape(src)}(?!\w)", flags=re.IGNORECASE)
                        replacements.append((pat, dst))

        # Step 2: normalize NL
        cleaned_src = []
        for text in self.src_lines:
            t = text.strip().lower()
            for pat, dst in replacements:
                t = pat.sub(dst.lower(), t)
            t = re.sub(r"\s+", " ", t)
            cleaned_src.append(t)
        self.src_lines = cleaned_src

        # Step 3: normalize SQL (for train/dev only)
        if split != "test" and self.tgt_lines is not None:
            cleaned_tgt = []
            for q in self.tgt_lines:
                q = q.strip()
                #if not q.endswith(";"):
                    #q += ";"
                q = re.sub(r"\s+", " ", q)
                cleaned_tgt.append(q)
            self.tgt_lines = cleaned_tgt
    
    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        input_text = f"translate English to SQL: {self.src_lines[idx]}"

        tokens = self.tokenizer(input_text, max_length=256, truncation=True, return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        if self.split != "test":
            labels = self.tokenizer(self.tgt_lines[idx], max_length=256, truncation=True, return_tensors="pt")
            label_ids = labels["input_ids"].squeeze(0)
            dictionary = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids, "bos_id": self.bos_id}
        else:
            dictionary = {"input_ids": input_ids, "attention_mask": attention_mask, "bos_id": self.bos_id}

        return dictionary

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                            the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # working on this at the moment

    encoder_ids = []
    encoder_masks = []
    decoder_inputs = []
    decoder_targets = []
    initial_decoder_inputs = []

    for ex in batch:
        encoder_ids.append(ex["input_ids"])
        encoder_masks.append(ex["attention_mask"])

        labels = ex["labels"]
        # trim to true (non-PAD) length so loss ignores trailing pads
        valid_len = int((labels != PAD_IDX).sum().item())
        valid_len = max(valid_len, 1)
        labels_trim = labels[:valid_len]

        bos_id = ex["bos_id"]
        # decoder_inputs = [BOS] + labels[:-1]
        di = torch.cat([torch.tensor([bos_id], dtype=torch.long), labels_trim[:-1]], dim=0)
        decoder_inputs.append(di)
        decoder_targets.append(labels_trim)

        # single-token start for generation paths
        initial_decoder_inputs.append(torch.tensor([bos_id], dtype=torch.long))

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_masks = pad_sequence(encoder_masks, batch_first=True, padding_value=0)
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = pad_sequence(initial_decoder_inputs, batch_first=True, padding_value=PAD_IDX)


    return encoder_ids, encoder_masks, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                            the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    
    encoder_ids = []
    encoder_masks = []
    initial_decoder_inputs = []

    for ex in batch:
        encoder_ids.append(ex["input_ids"])
        encoder_masks.append(ex["attention_mask"])
        initial_decoder_inputs.append(torch.tensor([ex["bos_id"]], dtype=torch.long))

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_masks = pad_sequence(encoder_masks, batch_first=True, padding_value=0)
    initial_decoder_inputs = pad_sequence(initial_decoder_inputs, batch_first=True, padding_value=PAD_IDX)

    return encoder_ids, encoder_masks, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x