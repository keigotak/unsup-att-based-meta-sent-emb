import os
import torch
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
from AbstractGetSentenceEmbedding import *
from HelperFunctions import get_device, get_now


class GetSentenceBertWordEmbedding(AbstractGetSentenceEmbedding):
    def __init__(self, model_name, device='cpu'):
        super().__init__()
        self.device = get_device(device)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.to(self.device)
        self.tokenizer = self.model.tokenizer
        self.tokenization_mode = 'subword'
        self.subword_pooling_method = 'avg'
        self.sentence_pooling_method = 'max' # max
        self.embeddings = {self.model_name: {}}

        self.tag = get_now()
        self.information_file = f'../results/sberts/info-{self.tag}.txt'

        self.embeddings_path = Path(f'./sentence_embeddings/{model_name}-{self.tokenization_mode}-{self.subword_pooling_method}-embeddings.txt')
        self.indexer_path = Path(f'./sentence_embeddings/{model_name}-{self.tokenization_mode}-{self.subword_pooling_method}-indexer.txt')

        self.cached_embeddings = {}
        if self.embeddings_path.exists():
            with self.embeddings_path.open('r') as f:
                for t in f.readlines():
                    lines = t.strip().split('\t')
                    self.cached_embeddings[lines[0]] = [list(map(float, l.split(' '))) for l in lines[1:]] # key is sentID
        else:
            self.embeddings_path.touch()

        self.sent_to_id = {}
        if self.indexer_path.exists():
            with self.indexer_path.open('r') as f:
                for t in f.readlines():
                    sentID, sentence = t.strip().split('\t')
                    self.sent_to_id[sentence] = sentID
        else:
            self.indexer_path.touch()

        self.sentence_id = len(self.sent_to_id)
        self.with_save_embeddings = True
        self.with_save_word_embeddings = True
        self.with_embedding_updating = False
        self.with_train_model = True
        if self.with_train_model:
            self.model.train()
            self.word_embeddings = {}
        else:
            self.model.eval()
            self.word_embeddings = self.load_model()

    def train(self):
        self.model.train()

    def get_ids(self, sent):
        ids_sent = self.model.tokenize(sent)
        if self.tokenization_mode == 'original':
            ids_sent.data['input_ids'] = torch.LongTensor(self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token] + sent.split(' ') + [self.tokenizer.sep_token])).unsqueeze(0)
            ids_sent.data['token_type_ids'] = self.tokenizer.convert_ids_to_tokens(ids_sent1.data['input_ids'][0])
            ids_sent.data['attention_mask'] = torch.ones_like(ids_sent.data['input_ids'])
        return ids_sent

    def process_subword(self, subword_list, embeddings):
        # aggregate subwords embeddings
        subword_aggregated_embeddings = []
        for i in range(-1, max(subword_list) + 1):
            subword_positions = [j for j, x in enumerate(subword_list) if x == i]
            # if the word are subworded
            if len(subword_positions) > 1:
                subword_embeddings = []
                for subword_position in subword_positions:
                    if type(embeddings[subword_positions[0]]) == np.ndarray:
                        subword_embeddings.append(torch.FloatTensor(embeddings[subword_position]).requires_grad_(False))
                    else:
                        subword_embeddings.append(embeddings[subword_position].requires_grad_(False))
                # subword pooling
                if self.subword_pooling_method == 'avg':
                    pooled_subword_embedding = torch.mean(torch.stack(subword_embeddings), dim=0)
                elif self.subword_pooling_method == 'max':
                    pooled_subword_embedding, _ = torch.max(torch.stack(subword_embeddings), dim=0)
                elif self.subword_pooling_method == 'head':
                    pooled_subword_embedding = subword_embeddings[0]
                subword_aggregated_embeddings.append(pooled_subword_embedding)
            else:
                if len(subword_positions) == 0:
                    if type(embeddings[subword_positions]) == np.ndarray:
                        subword_aggregated_embeddings.append(torch.zeros_like(torch.FloatTensor(embeddings[0])).requires_grad_(False))
                    else:
                        subword_aggregated_embeddings.append(torch.zeros_like(embeddings[0]).requires_grad_(False))
                else:
                    if type(embeddings[subword_positions[0]]) == np.ndarray:
                        subword_aggregated_embeddings.append(torch.FloatTensor(embeddings[subword_positions[0]]).requires_grad_(False))
                    else:
                        subword_aggregated_embeddings.append(embeddings[subword_positions[0]].requires_grad_(False))
        return torch.stack(subword_aggregated_embeddings, dim=0)

    def get_word_embedding(self, sentence, with_process_subwords=True):
        if sentence in self.word_embeddings.keys() and not self.with_train_model:
            return self.word_embeddings[sentence]
        if '�' in sentence:
            sentence = sentence.replace('� ', '')
        if 'o ̯ reĝ' in sentence:
            sentence = sentence.replace('o ̯ reĝ', '')
        enc = self.model.tokenizer(sentence).encodings[0]

        indexes, subwords, subword_ids = [], [], []
        index, subword, subword_id = [], [], []

        for i, (o1, o2) in enumerate(zip(enc.offsets, enc.offsets[1:])):
            if o1 == (0, 0):
                continue
            if o1 == o2:
                index.append(enc.ids[i])
                subword.append(enc.tokens[i])
                subword_id.append(enc.ids[i])
            else:
                if o1[1] == o2[0]:
                    index.append(enc.ids[i])
                    subword.append(enc.tokens[i])
                    subword_id.append(enc.ids[i])
                else:
                    index.append(enc.ids[i])
                    subword.append(enc.tokens[i])
                    subword_id.append(enc.ids[i])

                    indexes.append(index)
                    subwords.append(subword)
                    subword_ids.append(subword_id)

                    index, subword, subword_id = [], [], []
        flatten_indexes = [[j] * len(subwords[j]) for j in range(len(subwords))]
        flatten_indexes = [-1] + [i for index in flatten_indexes for i in index] + [len(subwords)]

        emb_sent1 = self.model.encode(sentence, output_value='token_embeddings')
        emb_sent1 = self.model.forward({'input_ids': torch.as_tensor([enc.ids], dtype=torch.long, device=self.device), 'attention_mask': torch.as_tensor([enc.attention_mask], dtype=torch.long, device=self.device)})
        emb_sent1 = emb_sent1['token_embeddings'].squeeze(0).cpu().detach().numpy()

        if with_process_subwords:
            if self.tokenization_mode == 'subword':
                emb_sent1 = self.process_subword(flatten_indexes, emb_sent1)
        else:
            emb_sent1 = torch.as_tensor(emb_sent1, dtype=torch.float, device=self.device)
        embedding = [emb_sent1.squeeze(0).tolist()[1: -1]] # 1, length, embedding_dim
        if sentence not in self.word_embeddings.keys() and not self.with_train_model:
            self.word_embeddings[sentence] = {'ids': indexes, 'tokens': subwords, 'embeddings': embedding}
            self.with_embedding_updating = True
        elif self.with_train_model:
            self.with_embedding_updating = True

        return {'ids': indexes, 'tokens': subwords, 'embeddings': embedding}

    def get_word_embeddings(self, sent1, sent2):
        ids, tokens, embedding = [], [], []
        for sent in [sent1, sent2]:
            rets = self.get_word_embedding(sent)
            ids.extend(rets['ids'])
            tokens.extend(rets['tokens'])
            embedding.extend(rets['embeddings'])

        return {'ids': ids, 'tokens': tokens, 'embeddings': embedding}

    def convert_ids_to_tokens(self, ids):
        return [self.tokenizer.convert_ids_to_tokens(_ids) for _ids in ids]

    def get_model(self):
        return self.model

    def set_model(self, model_name):
        self.model_name = model_name

    def batcher(self, params, batch):
        sentences = [' '.join(sent) for sent in batch]  # To reconstruct sentence from list of words

        sentence_embeddings = []
        for sentence in sentences:
            sentence_embedding = self.get_word_embedding(sentence)['embeddings'][0]
            if self.sentence_pooling_method == 'avg':
                sentence_embedding = torch.mean(torch.as_tensor(sentence_embedding, dtype=torch.float, device='cpu').requires_grad_(False), dim=0)
            elif self.sentence_pooling_method == 'max':
                sentence_embedding, _ = torch.max(torch.as_tensor(sentence_embedding, dtype=torch.float, device='cpu').requires_grad_(False), dim=0)
            sentence_embeddings.append(sentence_embedding)  # get token embeddings
            self.embeddings[model_name][sentence] = sentence_embedding.tolist()
        return torch.stack(sentence_embeddings).numpy()

    def save_information_file(self):
        information_file = Path(self.information_file)
        if not information_file.parent.exists():
            information_file.parent.mkdir(exist_ok=True)

        with Path(self.information_file).open('w') as f:
            f.write(f'source: {",".join(self.model_name)}\n')
            f.write(f'tokenization_mode: {self.tokenization_mode}\n')
            f.write(f'subword_pooling_method: {self.subword_pooling_method}\n')
            f.write(f'sentence_pooling_method: {self.sentence_pooling_method}\n')

    def set_tag(self, tag):
        self.tag = tag
        self.save_model_path = f'../models/sberts-{self.tag}.pkl'
        self.information_file = f'../results/sberts/info-{self.tag}.txt'

    def save_model(self):
        with Path(f'./{self.model_name}_{self.sentence_pooling_method}.pt').open('wb') as f:
            torch.save(self.word_embeddings, f)

    def load_model(self):
        if not self.with_train_model:
            path = Path(f'./{self.model_name}_{self.sentence_pooling_method}.pt')
            if path.exists():
                with Path(f'./{self.model_name}_{self.sentence_pooling_method}.pt').open('rb') as f:
                    return torch.load(f)
        return {}


class GetSentenceBertEmbedding(AbstractGetSentenceEmbedding):
    def __init__(self):
        super().__init__()
        self.model_names = ['bert-large-nli-stsb-mean-tokens', 'roberta-large-nli-stsb-mean-tokens']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_reset_output_file = False
        self.with_save_embeddings = True

    def get_model(self):
        self.model = SentenceTransformer(self.model_name)
        return self.model

    def set_model(self, model_name):
        self.model_name = model_name

    def batcher(self, params, batch):
        sentences = [' '.join(sent) for sent in batch]  # To reconstruct sentence from list of words
        sentence_embeddings = self.model.encode(sentences)  # get sentence embeddings
        for sentence, sentence_embedding in zip(batch, sentence_embeddings):
            self.embeddings[model_name][' '.join(sentence)] = sentence_embedding.tolist()
        return sentence_embeddings


if __name__ == '__main__':
    is_pooled = False
    if is_pooled:
        cls = GetSentenceBertEmbedding()
        for model_name in cls.model_names:
            print(model_name)
            cls.set_model(model_name)
            cls.single_eval(model_name)
            if cls.with_reset_output_file:
                cls.with_reset_output_file = False
