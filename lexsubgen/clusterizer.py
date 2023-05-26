import ast
import json
import os
from typing import Iterable, List, Union, Tuple, Dict
from collections import defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


class SubstituteClusterizer:
    def __init__(self,
                 n_clusters: Union[int, str] ='maxsil=range(2, 10)',
                 weighted_tfidf: bool = False,
                 use_idf: bool = False,
                 clusterizer: str = 'agglomerative',
                 metrics='cosine',
                 linkage='average'):
        """
        :param n_clusters: number of clusters or strategy for selecting number of clusters
            'fix' - clustering with a fixed number of clusters
            'maxsil' - clusterization to maximize silhouette score, number of clusters is selected from clusters range:
                'maxsil' - range(2, number of contexts)
                'maxsil=5' - range(2, 5)
                'maxsil=range(2, 9)' - clusters range
        :param metrics: parameter for AgglomerativeClustering
        :param linkage: parameter for AgglomerativeClustering
        """
        self.ncluster_strategy = None
        self.n_clusters = n_clusters
        self.weighted_tfidf = weighted_tfidf
        self.idf = use_idf
        self.cluster_alg = clusterizer
        self.metric = metrics
        self.linkage = linkage

    @property
    def n_clusters(self):
        return self._nclusters

    @n_clusters.setter
    def n_clusters(self, value: Union[int, str]):

        if isinstance(value, int):
            self.ncluster_strategy = 'fix'
            self._nclusters = value

        elif isinstance(value, str):
            cl_strategy = value.split('=')
            self.ncluster_strategy = cl_strategy[0]

            if len(cl_strategy) == 1:
                self._nclusters = None
            elif len(cl_strategy) == 2:
                ast_s = ast.parse(cl_strategy[1], mode='eval')
                ncl = eval(compile(ast_s, '', mode='eval'))
                if isinstance(ncl, int):
                    self._nclusters = range(2, ncl)
                elif isinstance(ncl, range):
                    self._nclusters = ncl

    @staticmethod
    def _unite_lang_substitutes(lang_subst_paths: Iterable[os.PathLike],
                                target: str,
                                n_subst: int = 3) -> Tuple[List[str], List[str]]:
        """
        Unites substitutes from different languages in one document
        :param lang_subst_paths: path to json file with substitutes ({target: {instance: substitutes str}}
        :param target: target word
        :param n_subst: how many substitutes of each language use for clustering
        """
        contexts = defaultdict(str)
        for lang in lang_subst_paths:
            with open(lang, 'r') as f:
                lang_subst = json.load(f)
            for context, subst in lang_subst[target].items():
                contexts[context] += ' '.join(subst.split()[:n_subst]) + ' '
        return list(contexts.keys()), list(contexts.values())

    @staticmethod
    def _transform4weighted(substitutes: List[str],
                            n_subst: int) -> List[str]:
        repeated_substitutes = []
        for context_subst in substitutes:
            context_subst = context_subst.split()
            repeated_substitutes.append(' '.join(
                [' '.join([context_subst[i]] * (n_subst - i % n_subst)) for i in range(len(context_subst))]))
        return repeated_substitutes

    def _vectorize(self,
                  documents: List[str]) -> np.ndarray:
        """
        Vectorizes documents for one target
        :param documents: list of string of substitutes for every context
        :return:
        """
        vectorizer = TfidfVectorizer(analyzer=lambda s: s.split(), use_idf=self.idf)
        vectorized_documents = vectorizer.fit_transform(documents)
        return vectorized_documents.toarray()

    def _perform_clustering(self, n_clusters: int, vectors: np.ndarray):
        if self.cluster_alg == 'agglomerative':
            clusterizer = AgglomerativeClustering(n_clusters,
                                                  metric=self.metric,
                                                  linkage=self.linkage)
        elif self.cluster_alg == 'kmeans':
            clusterizer = KMeans(n_clusters)

        clusterizer.fit(vectors)
        return clusterizer.labels_

    def _maximize_silscore(self, vectors: np.ndarray):
        scores = {}
        if not self.n_clusters:
            ncl_range = range(2, len(vectors))
        else:
            ncl_range = self.n_clusters
        for n in ncl_range:
            cl_labels = self._perform_clustering(n, vectors)
            scores[n] = silhouette_score(vectors, cl_labels, metric=self.metric)
        best_cl = max(scores.items(), key=lambda x: x[1])
        final_labels = self._perform_clustering(n_clusters=best_cl[0], vectors=vectors)
        return final_labels, best_cl[1]

    def clusterize_instances(self,
                             target_word: str,
                             lang_subst_path: Iterable[os.PathLike],
                             n_subst: int = 3) -> Tuple[float, List[Tuple[str, str]]]:
        context_ids, documents = self._unite_lang_substitutes(lang_subst_path, target_word, n_subst)
        if self.weighted_tfidf:
            documents = self._transform4weighted(substitutes=documents, n_subst=n_subst)
        vectors = self._vectorize(documents)
        if self.ncluster_strategy == 'fix':
            labels = self._perform_clustering(self.n_clusters, vectors)
            sil_score = silhouette_score(vectors, labels, metric=self.metric)
        elif self.ncluster_strategy == 'maxsil':
            labels, sil_score = self._maximize_silscore(vectors)

        return sil_score, list(zip(context_ids, labels))

    def cluster_all(self,
                    lang_subst_paths: Iterable[os.PathLike],
                    n_subst: int = 3):
        with open(lang_subst_paths[0], 'r', encoding='utf-8') as f:
            targets = list(json.load(f).keys())
        full_clustering = {}
        for target in tqdm(targets):
            res = self.clusterize_instances(target, lang_subst_paths, n_subst)
            full_clustering[target] = res
        return full_clustering

    def _save_clusterization(self,
                             clusterization: Dict[str, Tuple[float, List[Tuple[str, str]]]],
                             params: dict,
                             save_dir: Union[str, os.PathLike],
                             filename: str = None):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        if not filename:
            filename = 'clusterization.json'
        result_file = Path(save_dir, filename)

        class MyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return super(MyEncoder, self).default(obj)

        with open(result_file, 'w', encoding='utf-8') as newf:
            json.dump((params, clusterization), newf, cls=MyEncoder, default=str)
        return result_file