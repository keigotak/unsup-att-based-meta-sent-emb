from tqdm import tqdm
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import sys

import dask
import numpy as np
import torch
import torch.nn as nn

from scipy.stats import spearmanr, pearsonr
from senteval.utils import cosine

from GetSentenceBertEmbedding import GetSentenceBertWordEmbedding
from AbstractGetSentenceEmbedding import *
from AbstractTrainer import *
from ValueWatcher import ValueWatcher
from DataPooler import DataPooler
from HelperFunctions import set_seed, get_now, get_device

class VectorAttention(nn.Module):
    def __init__(self, model_names):
        super().__init__()
        set_seed(0)
        self.model_names = model_names
        self.model_dims = {'roberta-large-nli-stsb-mean-tokens': 1024, 'bert-large-nli-stsb-mean-tokens': 1024}
        self.embedding_dims = {model: self.model_dims[model] for model in self.model_names}
        self.meta_embedding_dim = 1024
        self.projection_matrices = nn.ModuleDict({model: nn.Linear(self.embedding_dims[model], self.meta_embedding_dim, bias=False) for model in self.model_names})
        self.max_sentence_length = 128
        self.vector_attention = nn.ModuleDict({model: nn.Linear(1, 1, bias=False) for model in self.model_names})
        self.normalizer = nn.ModuleDict({model: nn.LayerNorm([self.max_sentence_length, self.meta_embedding_dim]) for model in self.model_names})
        self.activation = nn.GELU()

class TrainVectorAttentionWithSTSBenchmark(AbstractTrainer):
    def __init__(self, device='cpu', model_names=None):
        self.device = get_device(device)
        if model_names is not None:
            self.model_names = model_names
        else:
            self.model_names = ['bert-large-nli-stsb-mean-tokens', 'roberta-large-nli-stsb-mean-tokens']
        self.va = VectorAttention(model_names=self.model_names).to(self.device)
        self.va.train()

        self.model_dims = self.va.model_dims
        self.source = self.get_source_embeddings()
        self.embedding_dims = {model: self.model_dims[model] for model in self.model_names}
        self.total_dim = sum(self.embedding_dims.values())

        self.tokenization_mode = self.source[self.model_names[0]].tokenization_mode
        self.subword_pooling_method = self.source[self.model_names[0]].subword_pooling_method
        self.source_pooling_method = 'concat' # avg, concat
        self.sentence_pooling_method = 'max' # avg, max

        self.gradient_clip = 0.0
        self.weight_decay = 1e-4
        self.with_vector_attention = True
        self.with_projection_matrix = True
        self.with_train_coefficients = True
        self.with_train_model = False
        self.loss_mode = 'word' # word, rscore

        self.alpha = nn.Linear(1, 1, bias=False).to(device=self.device)
        self.lam = nn.Linear(1, 1, bias=False).to(device=self.device)
        self.beta = nn.Linear(1, 1, bias=False).to(device=self.device)
        if self.with_train_coefficients:
            self.parameters = list(self.va.parameters()) + list(self.alpha.parameters()) + list(self.lam.parameters()) + list(self.beta.parameters())
        else:
            nn.init.constant_(self.alpha.weight, 1.0)
            nn.init.constant_(self.lam.weight, 1.0)
            nn.init.constant_(self.beta.weight, 1.0)
            self.parameters = list(self.va.parameters())

        if self.with_train_model:
            for model_name in model_names:
                self.parameters += list(self.source[model_name].model.parameters())

        self.learning_ratio = 1e-2

        super().__init__()

        self.batch_size = 512
        self.datasets_stsb['train'].batch_size = self.batch_size
        self.information_file = f'../results/vec_attention/info-{self.tag}.txt'
        self.vw.threshold = 5

    def get_source_embeddings(self):
        sources = {}
        for model in self.model_names:
            if model in set(['roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens']):
                sources[model] = GetSentenceBertWordEmbedding(model, device=self.device)
        return sources

    def batch_step(self, batch_embeddings, scores, with_training=False, with_calc_similality=False):
        running_loss = 0.0
        if with_training:
            if not self.va.training:
                self.va.train()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            if self.va.training:
                self.va.eval()

        gs_scores, sys_scores, losses = [], [], []
        padded_sequences, _ = self.modify_batch_embeddings_to_easy_to_compute(batch_embeddings)

        sentence_embeddings, word_embeddings = [], []
        for i in range(2):  # for input sentences, sentence1 and sentence2
            pooled_sentence_embedding, word_embedding = self.step({model_name: padded_sequences[model_name][i] for model_name in self.model_names})
            sentence_embeddings.append(pooled_sentence_embedding)
            word_embeddings.append(word_embedding)

        if self.loss_mode == 'word':
            # dimensions: sentence, source, words, hidden
            loss1, loss2, loss3 = [], [], []
            for word_embedding in word_embeddings:
                combinations = set()

                for iw1 in range(len(self.model_names)):
                    for iw2 in range(len(self.model_names)):
                        if iw1 == iw2 or (iw1, iw2) in combinations:
                            continue
                        combinations.add((iw1, iw2))
                        combinations.add((iw2, iw1))

                        words1 = word_embedding[self.model_names[iw1]]
                        words2 = word_embedding[self.model_names[iw2]]

                        sentence_length = words1.shape[1]
                        for i in range(sentence_length):
                            for j in range(sentence_length):
                                if i == j: # same sentence, same word
                                    loss1.append(torch.norm(words1[:, i] - words2[:, j], dim=1))
                                else: # same sentence, not same word
                                    loss2.append((-self.alpha.weight * torch.norm(words1[:, i] - words2[:, j], dim=1)).squeeze(0))

                        # different sentence
                        loss3.append((-self.beta.weight * (torch.norm(sentence_embeddings[0] - sentence_embeddings[1], dim=1))).squeeze(0))

            embedding_loss = [(self.lam.weight * torch.norm(self.va.projection_matrices[model_name].weight.T @ self.va.projection_matrices[model_name].weight - torch.eye(self.va.embedding_dims[model_name], device=self.device))).squeeze(0) for model_name in self.model_names]

            if self.lam == 0.0:
                loss = torch.mean(torch.stack(loss1)) + torch.mean(torch.stack(loss2)) + torch.mean(torch.stack(loss3))
            else:
                loss = torch.mean(torch.stack(loss1)) + torch.abs(torch.mean(torch.stack(loss2))) + torch.abs(torch.mean(torch.stack(loss3))) + torch.abs(torch.mean(torch.stack(embedding_loss)))

        elif self.loss_mode == 'rscore':
            loss = torch.abs(torch.norm(sentence_embeddings[0] - sentence_embeddings[1], dim=1)) - torch.as_tensor(scores, dtype=torch.float, device=self.device)
            loss = torch.mean(loss)

        if with_calc_similality:
            sys_score = [self.similarity(e1, e2) for e1, e2 in zip(sentence_embeddings[0].tolist(), sentence_embeddings[1].tolist())]
            sys_scores.extend(sys_score)
            gs_scores.extend(scores)

        running_loss += loss.item()
        print(running_loss)

        if with_training:
            loss.backward()
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.parameters, self.gradient_clip)
            self.optimizer.step()
            del loss

        return gs_scores, sys_scores, running_loss

    def step(self, feature, padding_mask=None):
        if self.with_projection_matrix:
            projected_embeddings = {
                model_name: self.va.projection_matrices[model_name](feature[model_name])
                for model_name in self.model_names
            }
        else:
            projected_embeddings = feature

        if self.with_vector_attention:
            pad_embeddings = {
                model_name: torch.cat((projected_embeddings[model_name],
                               torch.as_tensor([[[0.0] * self.va.meta_embedding_dim]
                                   * (self.va.max_sentence_length - projected_embeddings[model_name].shape[1])]
                                   * projected_embeddings[model_name].shape[0], dtype=torch.float, device=self.device)),
                    dim=1)
                for model_name in self.model_names}

            word_embeddings = {
                model_name: pad_embeddings[model_name] * self.va.vector_attention[model_name].weight.squeeze(0)
                for model_name in self.model_names}
            word_embeddings = {
                model_name: torch.narrow(word_embeddings[model_name], dim=1, start=0, length=feature[model_name].shape[1])
                for model_name in self.model_names}
        else:
            word_embeddings = projected_embeddings

        # multiple source embedding and vector attention
        if self.source_pooling_method == 'avg':
            pooled_word_embeddings = torch.mean(torch.stack([word_embeddings[model_name] for model_name in self.model_names]), dim=0)
        elif self.source_pooling_method == 'concat':
            try:
                pooled_word_embeddings = torch.cat([word_embeddings[model_name] for model_name in self.model_names], dim=2)
            except:
                print("shape error")

        # aggregate word embeddings to sentence embedding
        if self.sentence_pooling_method == 'avg':
            pooled_sentence_embedding = torch.mean(pooled_word_embeddings, dim=1)
        elif self.sentence_pooling_method == 'max':
            pooled_sentence_embedding, _ = torch.max(pooled_word_embeddings, dim=1)

        return pooled_sentence_embedding, word_embeddings

    def get_save_path(self, tag):
        return f'../models/vec_attention-{self.tag}-{tag}.pkl'

    def save_model(self):
        torch.save(self.va, self.get_save_path('va'))
        self.save_information_file()
        for model_name in self.model_names:
            if self.source[model_name].with_embedding_updating:
                with Path(f'./{model_name}.pt').open('wb') as f:
                    torch.save(self.source[model_name].word_embeddings, f)

    def load_model(self):
        if not os.path.exists(self.get_save_path('va')):
            pass
        else:
            self.va = torch.load(self.get_save_path('va'))
            self.va.to(self.device)
            print('\n'.join([f'{k}: {float(v.weight)}' for k, v in self.va.vector_attention.items()]))

    def save_information_file(self):
        super().save_information_file()

        with Path(self.information_file).open('w') as f:
            f.write(f'tag: {self.tag}\n')
            f.write(f'source: {",".join(self.model_names)}\n')
            f.write(f'meta_embedding_dim: {self.va.meta_embedding_dim}\n')
            f.write(f'tokenization_mode: {self.tokenization_mode}\n')
            f.write(f'subword_pooling_method: {self.subword_pooling_method}\n')
            f.write(f'source_pooling_method: {self.source_pooling_method}\n')
            f.write(f'sentence_pooling_method: {self.sentence_pooling_method}\n')
            f.write(f'learning_ratio: {self.learning_ratio}\n')
            f.write(f'gradient_clip: {self.gradient_clip}\n')
            f.write(f'weight_decay: {self.weight_decay}\n')
            f.write(f'alpha: {self.alpha.weight}\n')
            f.write(f'lambda: {self.lam.weight}\n')
            f.write(f'beta: {self.beta.weight}\n')
            f.write(f'batch_size: {self.batch_size}\n')
            f.write(f'with_vector_attention: {self.with_vector_attention}\n')
            f.write(f'with_projection_matrix: {self.with_projection_matrix}\n')
            f.write(f'with_train_coefficients: {self.with_train_coefficients}\n')
            f.write(f'with_train_model: {self.with_train_model}\n')
            f.write(f'dataset_type: {self.dataset_type}\n')
            f.write(f'loss_mode: {self.loss_mode}\n')
            f.write(f'weights: \n')
            f.write('\n'.join([f'{k}: {float(v.weight)}' for k, v in self.va.vector_attention.items()]))
            f.write('\n')
            f.write(str(self.optimizer))
            f.write('\n')


    def set_tag(self, tag):
        self.tag = tag
        self.information_file = f'../results/vec_attention/info-{self.tag}.txt'

    def update_hyper_parameters(self, hyper_params):
        self.va = VectorAttention(model_names=self.model_names).to(self.device)
        self.projection_matrices = self.va.projection_matrices
        self.vector_attention = self.va.vector_attention

        self.source_pooling_method = hyper_params['source_pooling_method']
        self.sentence_pooling_method = hyper_params['sentence_pooling_method']

        self.learning_ratio = hyper_params['learning_ratio']
        self.gradient_clip = hyper_params['gradient_clip']
        self.weight_decay = hyper_params['weight_decay']
        self.with_vector_attention = hyper_params['with_vector_attention']
        self.parameters = self.va.parameters()
        self.loss_mode = hyper_params['loss_mode']

        super().__init__()

        self.batch_size = hyper_params['batch_size']
        self.datasets_stsb['train'].batch_size = self.batch_size


class EvaluateVectorAttentionModel(AbstractGetSentenceEmbedding):
    def __init__(self, device):
        super().__init__()
        self.device = get_device(device)
        self.tag = get_now()
        self.model_names = ['bert-large-nli-stsb-mean-tokens', 'roberta-large-nli-stsb-mean-tokens']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_reset_output_file = False
        self.with_save_embeddings = True
        self.model = TrainVectorAttentionWithSTSBenchmark(device=device, model_names=self.model_names)
        self.model_tag = [f'vec_attention-{self.tag}']
        self.output_file_name = 'vec_attention.txt'
        self.model.va.eval()

    def get_model(self):
        return self.model

    def load_model(self):
        self.model.load_model()

    def batcher(self, params, batch):
        sentence_embeddings = []
        with torch.inference_mode():
            padded_sequences, padding_masks = self.modify_batch_sentences_for_senteval(batch)
            # get attention output
            sentence_embeddings, attention_weights = self.model.step({model_name: padded_sequences[model_name] for model_name in self.model_names},
                                                                   padding_mask={model_name: padding_masks[model_name] for model_name in self.model_names})

        return np.array(sentence_embeddings.tolist())

    def set_tag(self, tag):
        self.model_tag[0] = f'{self.model_tag[0]}-{tag}'
        self.tag = tag


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='select device')
    args = parser.parse_args()

    if args.device != 'cpu':
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    dp = DataPooler()
    es_metrics = 'pearson'
    if es_metrics == 'dev_loss':
        vw = ValueWatcher(mode='minimize')
    else:
        vw = ValueWatcher()
    cls = EvaluateVectorAttentionModel(device=args.device)
    trainer = cls.model
    trainer.model_names = cls.model_names
    trainer.set_tag(cls.tag)
    print(cls.tag)

    while not vw.is_over():
        print(f'epoch: {vw.epoch}')
        cls.model.train_epoch()
        cls.model.datasets_stsb['train'].reset(with_shuffle=True)
        rets = cls.model.inference(mode='dev')
        if es_metrics == 'pearson':
            vw.update(rets[es_metrics])
        else:
            vw.update(rets[es_metrics])
        if vw.is_updated():
            cls.model.save_model()
            dp.set('best-epoch', vw.epoch)
            dp.set('best-score', vw.max_score)
        dp.set(f'scores', rets)
    print(f'dev best scores: {cls.model.get_round_score(dp.get("best-score")[-1]) :.2f}')
    print(cls.model.information_file)

    cls.model.load_model()
    rets = cls.model.inference(mode='test')
    print(f'test best scores: ' + ' '.join(rets['prints']))
    rets = cls.single_eval(cls.model_tag[0])
    cls.model.append_information_file([f'es_metrics: {es_metrics}'])
    cls.model.append_information_file(rets['text'])
    print(cls.model.information_file)
