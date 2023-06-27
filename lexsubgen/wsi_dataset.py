import pandas as pd
import numpy as np
from torch.utils.data import Dataset


def simple_collate_fn(batch):
    batch_vector = np.array(batch)
    return batch_vector


class WSIDataset(Dataset):
    def __init__(self,
                 dataset: pd.DataFrame):
        self.target_lemmas = dataset['target_lemma'].to_numpy()
        self.context_ids = dataset['context_id'].to_numpy()
        self.target_ids = dataset['target_id'].to_numpy()
        self.tokenized_sentences = [sent.split() for sent in dataset['sentence']]

    def __getitem__(self, item):
        return [
            self.target_lemmas[item],
            self.target_ids[item],
            self.context_ids[item],
            self.tokenized_sentences[item]
        ]

    def __len__(self):
        return len(self.context_ids)