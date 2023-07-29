import os
from typing import List, Union, Tuple
import csv

from deep_translator import GoogleTranslator
import numpy as np
import spacy
import numpy.ma as ma
from simalign import SentenceAligner
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.stem.snowball import SnowballStemmer


class MultilingualSubstituteGenerator:

    nlp = spacy.load("en_core_web_sm")
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="f")
    stemmers = {
        'en': SnowballStemmer('english'),
        'fr': SnowballStemmer('french'),
        'es': SnowballStemmer('spanish'),
        'ru': SnowballStemmer('russian'),
        'de': SnowballStemmer('german'),
    }

    def __init__(self,
                 prob_estimator,
                 original_lang: str,
                 languages: List[str],
                 top_k: int = 20,
                 target_injection: bool = True,
                 source_lang_inj: bool = True,
                 words_only_subst: bool = True,
                 lemmatize: bool = True):

        self.prob_estimator = prob_estimator
        self.original_lang = original_lang
        self.top_k = top_k
        self.langs = languages
        self.source_lang_inj = source_lang_inj
        self.target_inj = target_injection
        self.words_only = words_only_subst
        self.lemmatize = lemmatize

    @staticmethod
    def translate_sentences(sentences: List[str],
                            from_lang: str,
                            to_lang: str) -> List[str]:
        translations = []
        translator = GoogleTranslator(from_lang, to_lang)
        for sent in sentences:
            translation = translator.translate(sent)
            translations.append(translation)
        return translations

    def align_targets(self,
                      source_sents: List[List[str]],
                      translations: List[List[str]],
                      target_posit: List[int]) -> List[int]:
        new_posits = []
        for sent, transl, posit in zip(source_sents, translations, target_posit):
            alignment = dict(self.aligner.get_word_aligns(sent, transl)['fwd'])
            if not alignment.get(posit):
                keys = list(alignment.keys())
                nearest = np.argmin(np.abs(np.array(keys) - posit))
                new = alignment[keys[nearest]]
            else:
                new = alignment[posit]
            new_posits.append(new)
        return new_posits

    @staticmethod
    def find_target_indexes(texts: List[str],
                            targets: List[str],
                            nlp: spacy) -> Tuple[List[List[str]], List[int]]:
        target_ids = []
        tokenized_texts = []
        for text, target in zip(texts, targets):
            tokenized_text = [t.text for t in nlp(text)]
            tokenized_texts.append(tokenized_text)
            for t in nlp(text):
                if t.lemma_ == target:
                    target_ids.append(t.i)
                    break
        return tokenized_texts, target_ids

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
                     masked_tokens: List[int] = None) -> List[List[int]]:
        if masked_tokens is not None:
            mask = np.zeros(probs.shape)
            mask[:, masked_tokens] = 1
            probs = ma.masked_array(probs, mask=mask)
        sorted_tokens = np.argsort(-probs, axis=-1)
        return sorted_tokens[:, :k].tolist()  # (n_sents, k)

    def _foreign_substitutes(self,
                             lang: str,
                             targets_ids: List[int],
                             sents: List[List[str]],
                             ):
        sent_str = [self.detokenizer.detokenize(sent) for sent in sents]
        translations = self.translate_sentences(sent_str, from_lang=self.original_lang, to_lang=lang)
        translations_tokenized = [self.tokenizer.tokenize(transl) for transl in translations]
        translated_targets_ids = self.align_targets(sents,
                                                    translations_tokenized,
                                                    targets_ids)
        return translations_tokenized, translated_targets_ids

    def generate_batch_substitutes(self,
                                   lang: str,
                                   targets: List[str],
                                   targets_ids: List[int],
                                   sents: List[List[str]],
                                   context_ids: List[Union[str, int]],
                                   ):
        batch_size = len(sents)
        if lang != self.original_lang:
            translations_tokenized, translated_targets_ids = self._foreign_substitutes(
                lang,
                targets_ids,
                sents,
            )
        else:
            translations_tokenized, translated_targets_ids = sents, targets_ids

        if self.target_inj:
            if self.source_lang_inj:
                targets2inject = [sents[i][targets_ids[i]]
                                  for i in range(batch_size)]
            else:
                targets2inject = [translations_tokenized[i][translated_targets_ids[i]]
                                  for i in range(batch_size)]
        else:
            targets2inject = None

        probs = self.get_probs(translations_tokenized,
                               translated_targets_ids,
                               targets2inject)
        if self.words_only:
            mask = self.prob_estimator.subword_mask
        else:
            mask = None
        substitutes_ids = self.select_top_k(probs, self.top_k, masked_tokens=mask)
        substitutes = [self.prob_estimator.tokenizer.convert_ids_to_tokens(sent_subst) for sent_subst in substitutes_ids]
        if self.lemmatize:
            substitutes = [[self.stemmers[lang].stem(s) for s in sent]for sent in substitutes ]
        substitutes = [' '.join(context_subst) for context_subst in substitutes]
        return [targets, context_ids, substitutes]

    def generate_all_langs(self,
                           targets: List[str],
                           targets_ids: List[int],
                           sents: List[List[str]],
                           context_ids: List[Union[str, int]],
                           ) -> List[Tuple[str, str, str, str]]:
        all_substitutes = []
        for lang in self.langs:
            substs = self.generate_batch_substitutes(lang, targets, targets_ids, sents, context_ids)
            lang_vector = [lang] * len(targets)
            all_substitutes.extend(list(zip(lang_vector, *substs)))
        return all_substitutes

    @staticmethod
    def write_substitutes(substitutes: List[Tuple[str, str, str, str]],
                          path: Union[str, os.PathLike]):
        with open(path, 'a', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(substitutes)
