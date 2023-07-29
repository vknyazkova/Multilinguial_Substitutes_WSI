import ast
import os
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Iterable, Dict, Tuple, Set, Union, Hashable

import fasttext
import numpy as np
import pandas as pd
import spacy
from deep_translator import GoogleTranslator
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def deep_translator_google(
        text: str,
        source: str,
        target: str
) -> str:
    return GoogleTranslator(source=source, target=target).translate(text)


class FasttextSubstituteGenerator:
    def __init__(self,
                 dataset: pd.DataFrame,
                 lang_model: fasttext.FastText,
                 subst_lang: str,
                 original_lang='en',
                 n_substitutes=200,
                 path_to_align_matrix: Union[str, os.PathLike] = None,
                 context_retriever: str = 'dummy'):
        """

        :param dataset: pandas DataFrame with information about every context, should include columns:
            sentence - string with parsed context
            target_id - idx of target in parsed context
            context_id - context id in semeval format (ex. access.n.20')
            column with target name
        :param lang_model: FastText language model
        :param subst_lang: iso language code for the language of the substitutes
        :param original_lang: iso language code for the language of the contexts
        :param n_substitutes: how many substitutes to generate
        :param path_to_align_matrix: path to txt with alignement matrix
        :param context_retriever:  how to retrieve context of the target:
            'dummy' - just take words from the left and from the right with window size
            'pos_excluding' - take words from the left and from the right and remove function words (like auxiliary verbs, articles and etc.)
        """

        self.dataset = dataset
        self.lang_model = lang_model

        self.source_lang = original_lang
        self.subst_lang = subst_lang
        self.n_subst = n_substitutes

        self.target2context2idx = None
        self.context_embs = None

        if path_to_align_matrix:
            self.alignment = np.loadtxt(path_to_align_matrix)
        else:
            self.alignment = None

        self.nlp = None
        self.context_retriever = context_retriever

    @property
    def context_retriever(self):
        return self._context_retriever

    @context_retriever.setter
    def context_retriever(self, name: str):
        if name == 'dummy':
            self._context_retriever = self._dummy_target_context
        elif name == 'pos_excluding':
            self.nlp = spacy.load('en_core_web_sm')
            self._context_retriever = self._pos_context
        else:
            assert ValueError('Wrong name for context_retriever. Should be either dummy or pos_excluding.')

    @staticmethod
    def _write_context_embs(context_embs: Dict[Hashable, np.ndarray],
                            target2context2idx: Dict[str, Dict[str, Tuple[int, str]]],
                            save_path: Union[str, os.PathLike]) -> None:
        """
        Writes context embeddings to json file

        :param context_embs: {target: target_context_embs}, target_context_embs.shape=(n_contexts, emb_size)
        :param target2context2idx: {target: {context_id: contex index in np.ndarray}}
        :param save_path: where to save context embeddings
        :return: None
        """

        context_embs = {target: embs.tolist() for target, embs in context_embs.items()}
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump((context_embs, target2context2idx), f)
        return None

    def _dummy_target_context(self,
                              context_info: pd.Series,
                              delete_target=True,
                              window=3) -> str:
        """
        Retrieves context with a given window size

        :param context_info: row in a dataframe with context information. should contain columns 'target_id' and 'sentence'
        :param delete_target: delete target or not from the context
        :return: context string
        """
        target_id = context_info['target_id']
        context = ast.literal_eval(context_info['sentence'])
        if delete_target:
            context.pop(target_id)
        text_context = ' '.join(context[target_id - window: target_id + window + 1])
        return re.sub(r'[^\w\s\-]', '', text_context).lower().strip()

    def _pos_context(self,
                     context_info: pd.Series,
                     delete_target=True,
                     window=10) -> str:
        """
        Retrieves context with a given window size excluding some pos-tags

        :param context_info: row in dataframe with context information. should contain columns 'target_id' and 'sentence'
        :param delete_target: delete target or not from the context
        :return: context string
        """
        to_delete_pos = ['PUNCT', 'DET', 'PART', 'X', 'AUX']

        full_context = ' '.join(ast.literal_eval(context_info['sentence']))
        parsed_context = self.nlp(full_context)

        target_id = context_info['target_id']
        window_context = parsed_context[target_id - window: target_id + window + 1]
        window_context = [t.lemma_ for t in window_context if t.pos_ not in to_delete_pos]

        if delete_target:
            try:
                window_context.remove(context_info['context_id'].split('.')[0])
            except ValueError:
                pass
        window_context_str = ' '.join(window_context).lower()
        if len(window_context) <= 1:
            window_context_str = self._pos_context(context_info, delete_target, window=window * 2)
        return window_context_str

    def get_context_embeddings(self,
                               source_lang_model: fasttext.FastText,
                               groupby='group_by',
                               window=3,
                               context_emb_path: os.PathLike = None) -> None:
        """
        :param source_lang_model: fasttext language model
        :param groupby: column name to group contexts by (column with target word info)
        :param window: how many words retrieve as a context for a target word
        :param context_emb_path: path where to save context_embs
        :return:
        """
        by_targets = self.dataset.groupby([groupby])

        context_embeddings = {}
        target2context2idx = defaultdict(dict)

        for target in by_targets.groups.keys():
            target_contexts_embs = []
            for i, context in enumerate(by_targets.groups[target]):
                context_info = self.dataset.loc[context]
                extracted_context = self._context_retriever(context_info, window=window)
                cont_emb = source_lang_model.get_sentence_vector(extracted_context)
                target_contexts_embs.append(cont_emb)
                target2context2idx[target][context_info['context_id']] = (i, extracted_context)
            if isinstance(self.alignment, np.ndarray):
                target_contexts_embs = np.matmul(np.array(target_contexts_embs), self.alignment)
            context_embeddings[target] = np.array(target_contexts_embs)

        self.context_embs = context_embeddings
        self.target2context2idx = target2context2idx

        if context_emb_path:
            self._write_context_embs(context_embeddings, target2context2idx, context_emb_path)
        return None

    def _read_context_embs(self,
                           context_embs_path: Union[str, os.PathLike]) -> None:
        """Reads contexts embeddings to context_embs attribute from json file"""

        with open(context_embs_path, 'r', encoding='utf-8') as f:
            context_embs, target2context2idx = json.load(f)
        context_embs = {target: np.array(embs) for target, embs in context_embs.items()}

        self.context_embs = context_embs
        self.target2context2idx = target2context2idx
        return None

    def delete_typos(self, token_list: Iterable[str],
                     typo_threshold: int) -> Set[str]:
        """Removes tokens from token_list, that are probably typos"""
        to_delete = set()
        # сортирую по частотности в словаре fasttext, тк опечатки скорее всего будут менее частотными,
        # чем правильное написание
        token_list = sorted(token_list,
                            key=lambda x: self.lang_model.get_word_id(x) if x in self.lang_model.words else 10 ** 10)
        for i in range(len(token_list)):
            for j in range(i + 1, len(token_list)):
                if fuzz.ratio(token_list[i], token_list[j]) > typo_threshold:
                    # убираем токен, у которого маленькое редакционное расстояние с предыдущим словом так как
                    # скорее всего это будет написанное с ошибкой предыдущее слово
                    to_delete.add(token_list[j])
        return set(token_list) - to_delete

    def clean_substitutes(self,
                          substitutes: List[str],
                          typo_threshold: int) -> Set[str]:
        """
        Deletes typos and unnecessary punctuation

        :param substitutes: list of substitute
        :param typo_threshold: tokens with similarity more than given threshold
            will be assumed as a typos of the same word
        """
        unique_substitutes = set()
        for t in substitutes:
            t = t.lower().split('.')[0]  # тк в fasttext в словаре может быть такое: 'say.However'
            unique_substitutes.add(t)
        cleaned_substitutes = self.delete_typos(list(unique_substitutes), typo_threshold)
        if '' in cleaned_substitutes:
            cleaned_substitutes.remove('')
        return cleaned_substitutes

    def dummy_substitutes(self,
                          target: str,
                          typo_threshold: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        """
        n_closest words to target with their embeddings (don't concern context)
        :param target: target word (without pos tag) for which to find nearest neighbours
        :param typo_threshold: the maximum similarity between tokens to treat them as different words and not typos
        :return: (target_substitutes_labels, substitutes_embeddings)
            target_substitutes_labels.shape = (n_subst, )
            substitutes_embeddings.shape = (n_subst, emb_size)
        """
        substitutes = self.lang_model.get_nearest_neighbors(target, k=self.n_subst)
        substitutes = [s[1] for s in substitutes]
        cleaned_substitutes = self.clean_substitutes(substitutes, typo_threshold)
        target_closest_subst, target_closest_emb = [], []
        for substitute in cleaned_substitutes:
            emb = self.lang_model.get_word_vector(substitute)
            target_closest_subst.append(substitute)
            target_closest_emb.append(emb)
        target_closest_emb = np.array(target_closest_emb)

        if isinstance(self.alignment, np.ndarray):
            target_closest_emb = np.matmul(target_closest_emb, self.alignment)
        return np.array(target_closest_subst), np.array(target_closest_emb)

    @staticmethod
    def _translate_targets(targets_list: Iterable[str],
                           from_lang: str,
                           to_lang: str,
                           translator=deep_translator_google) -> Dict[str, str]:
        """
        Translates targets words
        :param targets_list: list with targets in semeval format
        :param from_lang: source language in iso format
        :param to_lang: target language in iso format
        :param translator: function(text, source_lang, target_lang) -> str
        :return: {target: target translation}
        """

        text = '\n'.join([t.split('.')[0] for t in targets_list])
        translation = translator(text, from_lang, to_lang).lower()
        translated_targets = {}
        for trn, target in zip(translation.split('\n'), targets_list):
            translated_targets[target] = trn

        return translated_targets

    @staticmethod
    def _save_translations(translations: Dict[str, str],
                           to_lang: str,
                           save_dir: Union[str, os.PathLike]) -> None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        result_file = Path(save_dir, f'{to_lang}_target.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(translations, f)
        return result_file

    def generate(self,
                 similarity_metric=cosine_similarity,
                 translated_targets: Dict[str, str] = None) -> Dict[str, Dict[str, str]]:
        """
        Generates substitutes for every context in dataset in lang_model language and sorts them by closeness to context

        :param similarity_metric:
        :param translated_targets: {target: translation} '
        :return: {target: {context: substitutes_str}}
        """

        subst_dict = defaultdict(dict)

        if self.subst_lang == self.source_lang:
            translated_targets = {target: target.split('.')[0] for target in self.context_embs.keys()}
        elif not translated_targets:
            translated_targets = self._translate_targets(targets_list=list(self.context_embs.keys()),
                                                         from_lang=self.source_lang,
                                                         to_lang=self.subst_lang)

        for i, target in enumerate(tqdm(self.context_embs)):
            target_lemma = translated_targets[target]
            dummy_subst, dummy_subst_embs = self.dummy_substitutes(target_lemma)
            context_subst_sim = similarity_metric(self.context_embs[target], dummy_subst_embs)
            contexts = [c[0] for c in sorted(self.target2context2idx[target].items(), key=lambda x: x[1][0])]
            for i in range(context_subst_sim.shape[0]):
                subst_argsort = np.argsort(-context_subst_sim[i])
                sorted_substitutes = dummy_subst[subst_argsort]
                subst_dict[target][contexts[i]] = ' '.join(sorted_substitutes)
        return subst_dict

    @staticmethod
    def _save_substitutes(substitutes: Dict[str, Dict[str, str]],
                          lang: str,
                          save_dir: os.PathLike,
                          filename: str = None):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        if not filename:
            filename = f'{lang}_substitutes.json'
        result_file = Path(save_dir, filename)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(substitutes, f)
