import ast
import json
import os
from typing import Iterable, List, Union, Tuple, Dict
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import  sklearn
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
                 linkage='average',
                 save_models: bool = False):
        """
        :param n_clusters: number of clusters or strategy for selecting number of clusters
            'fix' - clustering with a fixed number of clusters
            'maxsil' - clusterization to maximize silhouette score, number of clusters is selected from clusters range:
                'maxsil' - range(2, number of contexts)
                'maxsil=5' - range(2, 5)
                'maxsil=range(2, 9)' - clusters range
        :param weighted_tfidf: consider substitute rank in vectorization or not
        :param use_idf: use inverse document frequency or not
        :param clusterizer: what clustering algorithm to use:
            agglomerative - for AgglomerativeClustering
            kmeans - for Kmeans
        :param metrics: parameter for AgglomerativeClustering only
        :param linkage: parameter for AgglomerativeClustering only
        """
        self.ncluster_strategy = None
        self.n_clusters = n_clusters
        self.weighted_tfidf = weighted_tfidf
        self.idf = use_idf
        self.cluster_alg = clusterizer
        self.metric = metrics
        self.linkage = linkage

        self.save_models = save_models
        self._vectorizers = {}  # {target: vocabulary for tf-idf}
        self._clusterizers = {}  # {target: cluster_centroids

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

    @property
    def vectorizers(self):
        return self._vectorizers

    @property
    def clusterizers(self):
        return self._clusterizers

    @staticmethod
    def _unite_lang_substitutes(langs: Iterable[str],
                                target_data: pd.DataFrame,
                                n_subst: int = 3) -> Tuple[List[str], List[str]]:
        """
        Unites substitutes from different languages in one document

        :param lang_subst_paths: path to json file with substitutes ({target: {instance: substitutes str}}
        :param target: target word
        :param n_subst: how many substitutes of each language use for clustering
        :return: context_ids, documents
        """

        def select_substitutes(row, k):
            return ' '.join([' '.join(subst.split()[:k]) for subst in row['substitutes']])

        res = target_data[
            target_data['lang'].isin(langs)
        ].groupby(['context_id']).apply(partial(select_substitutes, k=n_subst))
        return res.index.to_list(), res.to_list()

    @staticmethod
    def _transform4weighted(substitutes: List[str],
                            n_subst: int) -> List[str]:
        """
        Multiplies substitutes as many times as its reversed rank

        :param substitutes: list of length n_contexts with string of joined substitutes for the current context
        :param n_subst: how many substitutes of each language
        :return: list of joined (and reduplicated) substitutes for each context
        """
        repeated_substitutes = []
        for context_subst in substitutes:
            context_subst = context_subst.split()
            repeated_substitutes.append(' '.join(
                [' '.join([context_subst[i]] * (n_subst - i % n_subst)) for i in range(len(context_subst))]))
        return repeated_substitutes

    def _vectorize(self,
                  documents: List[str]):
        """
        Vectorizes documents of one target

        :param documents: list of string of substitutes for every context
        :return: vectorized documents
        """
        vectorizer = TfidfVectorizer(analyzer=lambda s: s.split(), use_idf=self.idf)
        vectorized_documents = vectorizer.fit_transform(documents)
        return vectorized_documents.toarray(), vectorizer

    def _perform_clustering(self, target: str, n_clusters: int, vectors: np.ndarray):
        """
        Performs clustering using one of the algorithms

        :param n_clusters: number of clusters for clusterization
        :param vectors: vectors to clusterize
        :return: cluster labels
        """
        if self.cluster_alg == 'agglomerative':
            clusterizer = AgglomerativeClustering(n_clusters,
                                                  metric=self.metric,
                                                  linkage=self.linkage)
        elif self.cluster_alg == 'kmeans':
            clusterizer = KMeans(n_clusters)

        clusterizer.fit(vectors)

        if self.cluster_alg == 'kmeans' and self.save_models:
            self._clusterizers[target] = clusterizer.cluster_centers_
        return clusterizer.labels_

    def _maximize_silscore(self, target:str, vectors: np.ndarray):
        """
        Find clusterization maximizing silhouette score

        :param vectors: vectors to clusterize
        :return: (labels, sil_score) for the best clusterization
        """
        scores = {}
        if not self.n_clusters:
            ncl_range = range(2, len(vectors))
        else:
            ncl_range = self.n_clusters
        for n in ncl_range:
            cl_labels = self._perform_clustering(target, n, vectors)
            scores[n] = silhouette_score(vectors, cl_labels, metric=self.metric)
        best_cl = max(scores.items(), key=lambda x: x[1])
        final_labels = self._perform_clustering(target, n_clusters=best_cl[0], vectors=vectors)
        return final_labels, best_cl[1]

    def clusterize_instances(self,
                             target: str,
                             rows: pd.DataFrame,
                             langs: List[str],
                             n_subst: int = 3) -> Tuple[float, List[Tuple[str, str]]]:
        context_ids, documents = self._unite_lang_substitutes(langs, rows, n_subst)
        if self.weighted_tfidf:
            documents = self._transform4weighted(substitutes=documents, n_subst=n_subst)
        vectors, vectorizer = self._vectorize(documents)
        if self.save_models:
            self._vectorizers[target] = vectorizer.vocabulary_
        if self.ncluster_strategy == 'fix':
            labels = self._perform_clustering(target, self.n_clusters, vectors)
            sil_score = silhouette_score(vectors, labels, metric=self.metric)
        elif self.ncluster_strategy == 'maxsil':
            labels, sil_score = self._maximize_silscore(target, vectors)

        return sil_score, list(zip(context_ids, labels))

    def cluster_all(self,
                    subst_dataset: pd.DataFrame,
                    langs: List[str],
                    n_subst: int = 3):
        targets_contexts = subst_dataset.groupby(['target_lemma']).groups
        full_clustering = {}
        for target in tqdm(targets_contexts):
            res = self.clusterize_instances(target, subst_dataset.loc[targets_contexts[target]], langs, n_subst)
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