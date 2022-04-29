import os
from tqdm import tqdm
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import dask
import numpy as np
import torch
import torch.nn as nn

from scipy.stats import spearmanr, pearsonr
from senteval.utils import cosine

from STSDataset import STSBenchmarkDataset, STSDataset
from GetHuggingfaceEmbedding import GetHuggingfaceWordEmbedding
from AttentionModel import MultiheadSelfAttentionModel, AttentionModel
from AbstractGetSentenceEmbedding import *
from ValueWatcher import ValueWatcher
from DataPooler import DataPooler
from HelperFunctions import set_seed, get_now


class AbstractTrainer:
    def __init__(self):
        set_seed(0)
        self.dataset_type = 'normal'
        self.datasets_stsb = {mode: STSBenchmarkDataset(mode=mode) for mode in ['train', 'dev', 'test']}
        self.datasets_sts = {mode: STSDataset(mode=mode) for mode in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']}
        self.optimizer = torch.optim.AdamW(self.parameters, lr=self.learning_ratio, weight_decay=self.weight_decay)
        self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=2)
        self.tag = get_now()
        self.vw = ValueWatcher()

    def batch_step(self, batch_embeddings, scores, tokens, with_training=False, with_calc_similality=False):
        raise NotImplementedError

    def step(self, feature):
        raise NotImplementedError

    def train_epoch(self, with_pbar=False):
        mode = 'train'
        if with_pbar:
            pbar = tqdm(total=self.datasets_stsb[mode].dataset_size)

        ## batch loop
        while not self.datasets_stsb[mode].is_batch_end():
            sentences1, sentences2, scores = self.datasets_stsb[mode].get_batch()

            ## get vector representation for each embedding and batch data
            with torch.inference_mode(): # this line must be allocate outside of self.batch_step scope
                batch_embeddings = []
                batch_tokens = []
                for sent1, sent2 in zip(sentences1, sentences2):
                    embeddings = {}
                    tokens = {}
                    for model_name in self.model_names:
                        rets = self.source[model_name].get_word_embeddings(sent1, sent2)
                        if False:
                            print('\t'.join([' '.join(items) for items in rets['tokens']]))
                        embeddings[model_name] = rets['embeddings']
                        tokens[model_name] = rets['tokens']
                    batch_embeddings.append(embeddings)  ## batch, embedding type, sentence source, sentence length, hidden size
                    batch_tokens.append(tokens)

            ## get attention output
            _, _, _ = self.batch_step(batch_embeddings, scores, with_training=True)

            if with_pbar:
                pbar.update(self.datasets_stsb[mode].batch_size)

            # print(str(self.datasets_stsb[mode]) + f' loss: {running_loss}')

        if with_pbar:
            pbar.close()

    def train(self, num_epoch=100):
        for i in range(num_epoch):
            self.train_epoch()
            if self.datasets_stsb['train'].batch_mode == 'fixed' and self.datasets_stsb['train'].current >= self.datasets_stsb['train'].dataset_size:
                self.datasets_stsb['train'].reset(with_shuffle=True)
            elif self.datasets_stsb['train'].batch_mode == 'full':
                self.datasets_stsb['train'].reset(with_shuffle=True)
            rets = self.inference('dev')

            self.vw.update(rets['pearson'][0])
            if self.vw.is_max():
                trainer.save_model()

    def inference(self, mode='dev'):
        running_loss = 0.0
        results = {}
        pearson_rs, spearman_rhos = [], []

        # batch loop
        sys_scores, gs_scores = [], []
        with torch.inference_mode():
            while not self.datasets_stsb[mode].is_batch_end():
                sentences1, sentences2, scores = self.datasets_stsb[mode].get_batch()

                # get vector representation for each embedding
                batch_embeddings = []
                batch_tokens = []
                for sent1, sent2 in zip(sentences1, sentences2):
                    embeddings = {}
                    for model_name in self.model_names:
                        rets = self.source[model_name].get_word_embeddings(sent1, sent2)
                        if False:
                            print('\t'.join([' '.join(items) for items in rets['tokens']]))
                        embeddings[model_name] = rets['embeddings']
                    batch_embeddings.append(embeddings)  ## batch, embedding type, sentence source, sentence length, hidden size
                    batch_tokens.append([sent1.split(' '), sent2.split(' ')])

                # get attention output
                gs, sys, loss = self.batch_step(batch_embeddings, scores, with_calc_similality=True)
                sys_scores.extend(sys)
                gs_scores.extend(gs)
                running_loss += loss
        pearson_rs = pearsonr(sys_scores, gs_scores)[0]
        spearman_rhos = spearmanr(sys_scores, gs_scores)[0]

        avg_pearson_r = np.average(pearson_rs)
        avg_spearman_rho = np.average(spearman_rhos)

        results = {'pearson': avg_pearson_r,
                   'spearman': avg_spearman_rho,
                   'nsamples': len(sys_scores),
                   'dev_loss': running_loss}

        print_contents = [f'STSBenchmark-{mode}',
                          f'pearson: {self.get_round_score(results["pearson"]) :.2f}',
                          f'spearman: {self.get_round_score(results["spearman"]) :.2f}']
        results['prints'] = print_contents

        print(f'[{mode}] ' + str(self.datasets_stsb[mode]) + f' loss: {running_loss}')
        print(' '.join(print_contents))

        self.datasets_stsb[mode].reset()

        return results

    def inference_sts(self, mode='STS12'):
        running_loss = 0.0
        results = {}
        pearson_rs, spearman_rhos = [], []

        # batch loop
        sys_scores, gs_scores, tag_sequence = [], [], []
        with torch.inference_mode():
            while not self.datasets_sts[mode].is_batch_end():
                sentences1, sentences2, scores, tags = self.datasets_sts[mode].get_batch()

                # get vector representation for each embedding
                batch_embeddings = []
                batch_tokens = []
                for sent1, sent2 in zip(sentences1, sentences2):
                    embeddings = {}
                    for model_name in self.model_names:
                        rets = self.source[model_name].get_word_embeddings(sent1, sent2)
                        if False:
                            print('\t'.join([' '.join(items) for items in rets['tokens']]))
                        embeddings[model_name] = rets['embeddings']
                    batch_embeddings.append(embeddings)  ## batch, embedding type, sentence source, sentence length, hidden size
                    batch_tokens.append([sent1.split(' '), sent2.split(' ')])

                # get attention output
                gs, sys, loss = self.batch_step(batch_embeddings, scores, with_calc_similality=True)
                sys_scores.extend(sys)
                gs_scores.extend(gs)
                tag_sequence.extend(tags)
                running_loss += loss
        pearson_rs = pearsonr(sys_scores, gs_scores)[0]
        spearman_rhos = spearmanr(sys_scores, gs_scores)[0]

        avg_pearson_r = np.average(pearson_rs)
        avg_spearman_rho = np.average(spearman_rhos)

        results = {'pearson': avg_pearson_r,
                   'spearman': avg_spearman_rho,
                   'nsamples': len(sys_scores),
                   'dev_loss': running_loss,
                   'sys_scores': sys_scores,
                   'gold_scores': gs_scores,
                   'tags': tag_sequence
                   }

        print_contents = [f'STSBenchmark-{mode}',
                          f'pearson: {self.get_round_score(results["pearson"]) :.2f}',
                          f'spearman: {self.get_round_score(results["spearman"]) :.2f}']
        results['prints'] = print_contents

        self.datasets_sts[mode].reset()

        return results

    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def get_round_score(self, score):
        return Decimal(str(score * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)

    def append_information_file(self, results):
        information_file = Path(self.information_file)
        if not information_file.parent.exists():
            information_file.parent.mkdir(exist_ok=True)

        with information_file.open('a') as f:
            f.write('\n'.join(results))
            f.write('\n')

    def save_information_file(self):
        information_file = Path(self.information_file)
        if not information_file.parent.exists():
            information_file.parent.mkdir(exist_ok=True)

    def set_tag(self, tag):
        raise NotImplementedError

    def modify_batch_embeddings_to_easy_to_compute(self, batch_embeddings):
        padded_sequences = {model_name: [] for model_name in self.model_names}
        padding_masks = {model_name: [] for model_name in self.model_names}
        try:
            for model_name in self.model_names:
                for i in range(2):
                    padded_sequences[model_name].append(
                        torch.nn.utils.rnn.pad_sequence([torch.as_tensor(items[i], dtype=torch.float, device=self.device)
                                                         for items in [embs[model_name]
                                                                       for embs in batch_embeddings]], batch_first=True)
                    )
                    max_sentence_length = max([len(items[i]) for items in [embs[model_name] for embs in batch_embeddings]])
                    padding_masks[model_name].append(
                        torch.as_tensor([[[False] * len(embs[model_name][i]) +
                                           [True] * (max_sentence_length - len(embs[model_name][i])) ]
                                          for embs in batch_embeddings], dtype=torch.bool, device=self.device).squeeze(1)
                    )
        except:
            print("error")
        return padded_sequences, padding_masks

