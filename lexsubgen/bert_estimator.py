from typing import List, Iterable, Tuple

import numpy as np
import torch
from torch.nn.functional import softmax
import transformers
from transformers import AutoTokenizer, BertForMaskedLM


class BertProbEstimator:
    def __init__(self,
                 model_name: str = 'bert-base-multilingual-uncased',
                 target_injection: bool = True,
                 temperature: float = 1.0,
                 sim_metric: str = 'dot-product',
                 device: str = 'cpu'):
        self.model_name = model_name
        self.target_inj = target_injection
        self.temp = temperature
        self.sim_metric = sim_metric
        self.device = device

        self.__load_model()

    def _extract_embeddings(self):
        return self.model.bert.embeddings.word_embeddings.weight.data.cpu()  # (vocab_size, emb_size)

    def __load_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = BertForMaskedLM.from_pretrained(self.model_name)
        self._model.to(self.device).eval()
        self._embeddings = self._extract_embeddings()

        vocab = self._tokenizer.get_vocab()
        subword_indexes = [vocab[t] for t in vocab if t.startswith('##')]
        self._subword_mask = np.zeros(self._embeddings.size()[0])
        self._subword_mask[subword_indexes] = 1

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def subword_mask(self):
        return self._subword_mask

    def masked_target_tokenize(self,
                               sentences: Iterable[List[str]],
                               target_position: List[int]
                               ) -> Tuple[
                                  transformers.tokenization_utils_base.BatchEncoding,
                                  torch.tensor]:
        """
        Replaces word on target position with [MASK] and tokenizes using bert tokenizer
        Args:
            sentences: list of tokenized sentences
            target_position: list of indexes of the targets

        Returns:
            bert_tokized
            mask_posit: (n_sents, 2), 2 - sent_id, mask_id
        """

        for i, target_idx in enumerate(target_position):
            sentences[i][target_idx] = '[MASK]'
        bert_tokenized = self.tokenizer([' '.join(sent) for sent in sentences],
                                        padding='longest', return_tensors="pt")
        mask_posit = (bert_tokenized.input_ids == self.tokenizer.mask_token_id).nonzero()
        return bert_tokenized, mask_posit

    def context_conditioned_probs(self,
                                  tokenized_sentences: List[List[str]],
                                  target_position: List[int]) -> np.ndarray:
        """
        Computes log-probabilities over vocabulary with respect to context (P(s|C))
        Args:
            tokenized_sentences: list of simply tokenized sentences
            target_position: list of target indexes in sentences

        Returns:
            probs: (n_sents, vocab_size)
        """
        bert_tokenized, mask_posit = self.masked_target_tokenize(tokenized_sentences, target_position)

        with torch.no_grad():
            logits = self.model(**bert_tokenized).logits  # (n_sents, longest_seq, vocab_size)
        logits = logits[mask_posit[:, 0], mask_posit[:, 1], :]  # (n_sents, vocab_size)

        return softmax(logits, dim=-1).numpy()

    def _oov_embs(self, oov_words: List[str]) -> np.ndarray:
        """
        Calculates embeddings for out of vocabulary words as mean vector of the subwords
        Args:
            oov_words: list of oov words

        Returns:
            oov_embeddings: (n_oov_words, emb_size)
        """
        oov_embeddings = []
        for word in oov_words:
            sub_token_ids = self.tokenizer.encode(word)[1:-1]
            mean_vector = self.embeddings[sub_token_ids, :].mean(axis=0, keepdims=True).detach().cpu().numpy()
            oov_embeddings.append(mean_vector)
        return np.concatenate(oov_embeddings, axis=0)

    def target_conditioned_probs(self,
                                 targets: List[str]) -> np.ndarray:
        """
        Computes probabilities over vocabulary as closest to target
        Args:
            targets: list of targets

        Returns:
            probs: (n_sents, vocab_size)
        """
        targets = np.array(targets)
        target_ids = np.array(self.tokenizer.convert_tokens_to_ids(targets))
        target_embeddings = self.embeddings[target_ids, :]  # (n_instances, emb_size)

        oov_idx = np.where(target_ids == 100)[0]
        if np.any(oov_idx):
            oov_embeddings = self._oov_embs(targets[oov_idx])
            target_embeddings[oov_idx] = oov_embeddings

        if self.sim_metric == 'dot-product':
            logits = np.matmul(target_embeddings, self.embeddings.T)  # (n_instances, vocab_size)
        else:
            raise ValueError('Wrong similarity metric name')

        return softmax((logits / self.temp), dim=-1).numpy()