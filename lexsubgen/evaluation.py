import csv
import os
import re
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd

MATCH_SEMEVAL_SCORES_RE = re.compile(r"(\w+|\w+\.\w+)(\t*-?\d+\.?\d*)+")


def _compute_semeval_2013_metrics(
        java_file: os.PathLike,
        gold_labels_path: os.PathLike,
        pred_labels_path: os.PathLike
):
    scores = dict()
    output = subprocess.run(
        ["java", "-jar", java_file, gold_labels_path, pred_labels_path, 'all'],
        capture_output=True
    )
    for line in output.stdout.decode().split("\n"):
        if MATCH_SEMEVAL_SCORES_RE.match(line):
            word, *values = line.split('\t')
            scores[word] = tuple(float(val)*100.0 for val in values if val)
    return scores


def _convert_labels_to_semeval_format(
        target_words: List,
        context_ids: List,
        labels: List,
        save_path: os.PathLike
):
    """
    target_words, context_ids and labels have to be the same size = number of contexts

    :param target_words: target word for every context
    :param context_ids:context_id for every context
    :param labels: cluster (or sense) label for every context
    :param save_path: where to save semeval formatted file
    """
    for i in range(len(target_words)):
        if not context_ids[i].startswith(target_words[i]):
            context_ids[i] = f"{target_words[i]}.{context_ids[i]}"
    with open(save_path, "w") as fd:
        writer = csv.writer(fd, delimiter=" ")
        writer.writerows(zip(target_words, context_ids, labels))


def compute_semeval_2010_metrics(
    gold_labels: List,
    pred_labels: List,
    group_by: List[str],
    context_ids: List[str],
    data_path: os.PathLike,
    gold_labels_path: os.PathLike = None,
    ) -> Dict[str, List[float]]:

    with tempfile.TemporaryDirectory() as temp_directory:
        save_path = Path(temp_directory)
        pred_labels_path = save_path / "predicted-labels.key"
        _convert_labels_to_semeval_format(
            group_by, context_ids, pred_labels, pred_labels_path
        )
        if gold_labels_path is None:
            gold_labels_path = save_path / "gold-labels.key"
            _convert_labels_to_semeval_format(
                group_by, context_ids, gold_labels, gold_labels_path
            )

        fscore = _compute_semeval_2013_metrics(
            Path(data_path) / "evaluation" / "unsup_eval" / "fscore.jar",
            gold_labels_path,
            pred_labels_path,
        )
        vmeasure = _compute_semeval_2013_metrics(
            Path(data_path) / "evaluation" / "unsup_eval" / "vmeasure.jar",
            gold_labels_path,
            pred_labels_path,
        )

    metrics = dict()
    for word, (fs, prec, rec) in fscore.items():
        vm, homogeneity, completeness = vmeasure[word]
        metrics[word] = [fs, prec, rec, vm, homogeneity, completeness, (fs * vm) ** 0.5]
    print('fscore\tprecision\trecall\tvmeasure\thomogenity\completness')
    return metrics


class WSIEvaluator:
    def __init__(self,
                 dataset: pd.DataFrame,
                 clust_res_path: os.PathLike,
                 semeval_data_path: os.PathLike):
        self.dataset = dataset
        self.clust_res_path = clust_res_path
        self.semeval_folder = semeval_data_path

    def _read_cluster_labels(self):
        with open(self.clust_res_path, 'r') as f:
            clusterization_result = json.load(f)
        my_labels = {context[0]: '.'.join(context[0].split('.')[:2]) + '.' + context[1] for word in
                     clusterization_result for context in clusterization_result[word][1]}
        my_labels = [my_labels[idx] for idx in self.dataset["context_id"]]
        return my_labels

    def _read_gold_labels(self):
        gold_labels = dict()
        gold_labels_path = Path(self.semeval_folder) / "evaluation" / "unsup_eval" / "keys" / "all.key"
        with open(gold_labels_path, 'r') as f:
            for instance in f:
                _, context_id, *clusters = instance.strip().split(" ")
                # TODO: gold labels might consist of more than 1 cluster label
                gold_labels[context_id] = clusters[0]
        gold_labels = [gold_labels[idx] for idx in self.dataset["context_id"]]
        return gold_labels

    def compute_metrics(self):
        gold_labels = self._read_gold_labels()
        my_labels = self._read_cluster_labels()
        res = compute_semeval_2010_metrics(gold_labels=gold_labels,
                                           pred_labels=my_labels,
                                           group_by=self.dataset["group_by"].to_list(),
                                           context_ids=self.dataset["context_id"].to_list(),
                                           data_path=self.semeval_folder)
        metrics_names = ['fscore', 'precision', 'recall', 'vmeasure', 'homogenity', 'completeness', '(fs * vm) ** 0.5']
        scores = pd.DataFrame(res.values(),
                              columns=metrics_names,
                              index=res.keys())
        return scores
