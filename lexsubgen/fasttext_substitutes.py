import ast
import os
from collections import defaultdict
from typing import List, Iterable, Dict, Tuple, Set
from pathlib import Path
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from fuzzywuzzy import fuzz
from googletrans import Translator
from enum import Enum
from sklearn.metrics.pairwise import cosine_similarity
import fasttext


def get_unofficial_google_translate(
        text: str,
        source_lang: Enum,
        target_lang: Enum
) -> str:
    """
    We advise not to translate using this function in multiprocessing manner.
    """
    return Translator().translate(text, src=source_lang, dest=target_lang).text


class FasttextSubstituteGenerator:
    def __init__(self,
                 dataset: pd.DataFrame,
                 source_lang='en',
                 n_substitutes=200):
        """
        :param dataset: pandas dataframe with info for every context
        :param source_lang:
        """
        self.dataset = dataset
        self.source_lang = source_lang
        self.n_subst = n_substitutes
        self.context_embs = None  # {target: {context_id: embedding}}

    @staticmethod
    def _write_context_embs(context_embs: Dict[str, Dict[str, np.ndarray]],
                            save_path: os.PathLike) -> None:
        """Writes context embeddings to json file"""
        to_write_cont_embs = defaultdict(dict)
        for t in context_embs:
            for s in context_embs[t]:
                to_write_cont_embs[t][s] = context_embs[t][s].tolist()
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(to_write_cont_embs, f)
        return

    def _extract_target_context(self,
                                context_id: int,
                                delete_target=True,
                                window=3) -> str:
        """
        Retrieves context with a given window size

        :param context_id: row id in dataset for current context info
        :param delete_target: delete target or not from context
        """

        context_info = self.dataset.loc[context_id]
        target_id = context_info['target_id']
        context = ast.literal_eval(context_info['sentence'])
        if delete_target:
            context.pop(target_id)
        return ' '.join(context[target_id - window: target_id + window + 1])

    def get_context_embeddings(self,
                               source_lang_model: fasttext.FastText._FastText,
                               groupby='groupby',
                               window=3,
                               context_emb_path: os.PathLike = None) -> None:
        """

        :param groupby: column name to group contexts by (target word)
        :param window: how many words retrieve as a context for a target word
        :param source_lang_model:
        :param context_emb_path: path where to save context_embs
        :return:
        """
        by_targets = self.dataset.groupby([groupby])
        context_embeddings = defaultdict(dict)
        for target in by_targets.groups:
            for context in by_targets.groups[target]:
                context_info = self.dataset.loc[context]
                extracted_context = self._extract_target_context(context_info, window=window)
                cont_emb = source_lang_model.get_sentence_vector(extracted_context)
                context_embeddings[target][context_info['context_id']] = cont_emb
        self.context_embs = context_embeddings
        if context_emb_path:
            self._write_context_embs(context_embeddings, context_emb_path)
        return None

    def _read_context_embs(self, context_embs_path: os.PathLike) -> None:
        """Reads contexts embeddings to context_embs attribute from json file"""
        with open(context_embs_path, 'r', encoding='utf-8') as f:
            read_context_embs = json.load(f)
        context_embs = defaultdict(dict)
        for t in read_context_embs:
            for s in read_context_embs[t]:
                context_embs[t][s] = np.array(read_context_embs[t][s])
        self.context_embs = context_embs
        return

    def _translate_targets(self, to_lang: str,
                           translator=get_unofficial_google_translate,
                           save_dir: os.PathLike = None) -> Dict[str, str]:
        """
        Translates targets words to to_lang language
        :param to_lang: target language for translation
        :param translator: function(text, source_lang, target_lang) -> str
        :param save_translation_path: path where to save dict of translated words
        :return: {target: target translation}
        """
        targets_list = list(self.context_embs.keys())
        text = ''
        for target in targets_list:
            word, tag = target.split('.')
            if tag == 'v':
                word = 'to ' + word
            text += word + '\n'
        translation = translator(text, source_lang=self.source_lang, target_lang=to_lang)
        translated_targets = {}
        for trn, target in zip(translation.split('\n'), targets_list):
            translated_targets[target] = trn
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            result_file = Path(save_dir, f'{to_lang}_target.json')
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(translated_targets, f)
            print(result_file)
        return translated_targets

    @staticmethod
    def _read_target_translations(path_to_file: os.PathLike):
        with open(path_to_file, 'r', encoding='utf-8') as f:
            translated_targets = json.load(f)
        return translated_targets

    @staticmethod
    def delete_typos(token_list: Iterable[str],
                     lang_model: fasttext.FastText._FastText,
                     typo_threshold: int) -> Set[str]:
        """Removes tokens from token_list, that are probably typos"""
        to_delete = set()
        # сортирую по частотности в словаре fasttext, тк опечатки скорее всего будут менее частотными,
        # чем правильное написание
        token_list = sorted(token_list, key=lambda x: lang_model.words.index(x) if x in lang_model.words else 10 ** 10)
        for i in range(len(token_list)):
            for j in range(i + 1, len(token_list)):
                if fuzz.ratio(token_list[i], token_list[j]) > typo_threshold:
                    # убираем токен, у которого маленькое редакционное расстояние с предыдущим словом так как
                    # скорее всего это будет написанное с ошибкой предыдущее слово
                    to_delete.add(token_list[j])
        return set(token_list) - to_delete

    def clean_substitutes(self,
                          substitutes: List[str],
                          lang_model: fasttext.FastText._FastText,
                          typo_threshold: int) -> Set[str]:
        """
        Deletes typos and unnecessary punctuation

        :param substitutes: list of substitute
        :param lang_model: fasttext language model
        :param typo_threshold: tokens with similarity more than given threshold
            will be assumed as a typos of the same word
        """
        unique_substitutes = set()
        for t in substitutes:
            t = t.lower().split('.')[0]  # тк в fasttext в словаре может быть такое: 'say.However'
            unique_substitutes.add(t)
        cleaned_substitutes = self.delete_typos(list(unique_substitutes), lang_model, typo_threshold)
        if '' in cleaned_substitutes:
            cleaned_substitutes.remove('')
        return cleaned_substitutes

    def dummy_substitutes(self,
                          target: str,
                          lang_model: fasttext.FastText._FastText,
                          typo_threshold: int = 80) -> List[Tuple[str, np.ndarray]]:
        """
        n_closest words to target with their embeddings (don't concern context)
        """

        substitutes = lang_model.get_nearest_neighbors(target, k=self.n_subst)
        substitutes = [s[1] for s in substitutes]
        cleaned_substitutes = self.clean_substitutes(substitutes, lang_model, typo_threshold)
        target_closest_emb = [(substitute, lang_model.get_word_vector(substitute)) for substitute in
                              cleaned_substitutes]
        return target_closest_emb

    def generate(self,
                 lang_model: fasttext.FastText._FastText,
                 lang: str,
                 similarity_metric=cosine_similarity,
                 translated_targets: Dict[str, str] = None) -> Dict[str, Dict[str, str]]:
        """
        Generates substitutes for every context in dataset in lang_model language and sorts them by closeness to context
        :param lang_model: fasttext language model object
        :param similarity_metric:
        :param translated_targets: {target: translation} '
        :return: {target: {context: substitutes_str}}
        """

        subst_dict = defaultdict(dict)
        if not translated_targets:
            translated_targets = self._translate_targets(lang)

        for i, target in enumerate(tqdm(self.context_embs)):
            target_lemma = translated_targets[target]
            dummy_subst_embs = self.dummy_substitutes(target_lemma, lang_model)
            for context, context_emb in self.context_embs[target].items():
                sorted_substitutes = sorted(dummy_subst_embs,
                                            key=lambda x: similarity_metric(
                                                x[1].reshape(1, -1),
                                                context_emb.reshape(1, -1)
                                            ),
                                            reverse=True)
                substitutes = [s[0] for s in sorted_substitutes]
                subst_dict[target][context] = ' '.join(substitutes)
        return subst_dict

    @staticmethod
    def _save_substitutes(substitutes: Dict[str, Dict[str, str]],
                          save_dir: os.PathLike,
                          lang: str):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        result_file = Path(save_dir, f'{lang}_substitutes.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(substitutes, f)
