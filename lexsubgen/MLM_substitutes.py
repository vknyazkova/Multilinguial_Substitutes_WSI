from typing import List, Union, Dict, Iterable, Tuple

from deep_translator import GoogleTranslator
import numpy as np
import numpy.ma as ma
from simalign import SentenceAligner
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer


def top_k_strategy(probs: np.ndarray, k: int) -> List[List[int]]:
    """
    Function that implements top-k strategy, i.e. chooses k substitutes with highest probabilities.

    Args:
        probs: probability distribution
        k: number of top substitutes to take

    Returns:
        list of chosen indexes
    """
    parted = np.argpartition(probs, kth=range(-k, 0), axis=-1)
    sorted_ids = parted[:, -k:][:, ::-1]
    return sorted_ids.tolist()


class MultilingualSubstituteGenerator:

    nlp = spacy.load("ru_core_news_sm")

    def __init__(self,
                 prob_estimator,
                 original_lang: str,
                 languages: List[str],
                 top_k: int = 20,
                 target_injection: bool = True,
                 source_lang_inj: bool = True,
                 words_only_subst: bool = True):
        self.prob_estimator = prob_estimator
        self.original_lang = original_lang
        self.top_k = top_k
        self.source_lang_inj = source_lang_inj
        self.target_inj = target_injection
        self.words_only = words_only_subst
        self.aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="a")
        # self.tokenizer = TreebankWordTokenizer()

    @staticmethod
    def translate_sentences(sentences: Iterable[str],
                            from_lang: str,
                            to_lang: str):
        sentences = '\n'.join(sentences)
        translations = GoogleTranslator(from_lang, to_lang).translate(sentences)
        return translations.split('\n')

    @staticmethod
    def basic_tokenize(sentences: List[str]):
        tokenizer = TreebankWordTokenizer()
        basic_tokenized_sents = [tokenizer(s) for s in sentences]
        return basic_tokenized_sents

    def align_targets(self,
                      source_sents: List[List[str]],
                      translations: List[List[str]],
                      target_posit: List[int]):
        new_posits = []
        for sent, transl, posit in zip(source_sents, translations, target_posit):
            alignment = dict(self.aligner(sent, transl)['inter'])
            new_posits.append(alignment[posit])
        return new_posits

    def find_target_indexes(self,
                            sent: List[str],
                            target: List[str]) -> List[int]:
        raise NotImplementedError

    def get_probs(self,
                  sents: List[List[str]],
                  target_ids: List[int],
                  target2inject: List[str] = None):
        probs = self.prob_estimator.context_conditioned_probs(sents, target_ids)
        if self.target_inj and target2inject:
            target_closest = self.prob_estimator.target_conditioned_probs(target2inject)
            probs = probs * target_closest
        return probs  # (n_sents, vocab_size)

    @staticmethod
    def select_top_k(probs: np.ndarray,
                     k: int,
                     masked_tokens: List[int] = None):
        if masked_tokens:
            mask = np.zeros(probs.shape)
            mask[:, masked_tokens] = 1
            probs = ma.masked_array(probs, mask=mask)
        sorted_probs = np.sort(probs, axis=-1)
        return sorted_probs[:, :k]  # (n_sents, k)

    def generate_batch_substitutes(self,
                                   sents: List[Tuple[str, str]],
                                   targets: List[Union[str, int]],
                                   lang: str):
        """

        Args:
            sents: list of tuples (context label? context sentence)
            targets: list of either targets strings or targets position
            lang: ISO language - in what language generate substitutes
        Returns:

        """
        context_ids, sents = zip(*sents)
        translations = self.translate_sentences(sents, from_lang=self.original_lang, to_lang=lang)
        if isinstance(targets[0], str):
            targets = self.find_target_indexes(sents, targets)
        translations_tokenized = self.basic_tokenize(translations)
        original_tokenized = self.basic_tokenize(sents)

        translated_targets_ids = self.align_targets(original_tokenized,
                                                    translations_tokenized,
                                                    targets)
        if self.target_inj:
            if self.source_lang_inj:
                targets2inject = [original_tokenized[i] for i in targets]
            else:
                targets2inject = [translations_tokenized[i] for i in translated_targets_ids]
        else:
            targets2inject = None

        probs = self.get_probs(translations_tokenized,
                               translated_targets_ids,
                               targets2inject)

        candidates_ids = top_k_strategy(probs, self.top_k)
        candidates = [self.prob_estimator.tokenizer.convert_ids_to_tokens(sent_subst) for sent_subst in candidates_ids]
        return candidates




