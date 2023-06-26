from functools import partial

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# def pad_sents(batch, tokenizer, out_voc, pad_value=1):
#     input, output = zip(*batch)
#
#     input = pad_sequence(input, batch_first=True, padding_value=pad_value)
#     output = pad_sequence(output, batch_first=True, padding_value=pad_value)
#     return input, output
#
# pad_collate_fn = partial(pad_sents, inp_voc=inp_voc, out_voc=out_voc, pad_value=1)


class WSIDataset(Dataset):
    def __init__(self,
                 dataset: pd.DataFrame):
        self.target_labels = dataset['target_labels']
        self.context_ids = dataset['context_id']
        self.tokenized_sentences = [sent.split() for sent in dataset['sentence']]

    def __getitem__(self, item):
        return {
            'target_label': self.target_labels[item],
            'context_label': self.context_ids[item],
            'sentence': self.tokenized_sentences[item]
        }

    def __len__(self):
        return len(self.context_ids)