import os
import pickle
from decimal import Decimal, ROUND_HALF_UP

import torch

import senteval


class AbstractGetSentenceEmbedding:
    def __init__(self):
        self.model_names = None
        self.embeddings = None
        self.model = None
        self.output_file_name = 'results.txt'
        self.with_detailed_log = False
        self.with_reset_output_file = False
        self.with_save_embeddings = False

    def get_model(self):
        pass

    def batcher(self, params, batch):
        return

    def eval(self):
        for model_name in self.model_names:
            self.single_eval(model_name=model_name)

    def get_params(self):
        return {'task_path': '/clwork/keigo/SentenceMetaEmbedding/data', 'usepytorch': True, 'batch_size': 10000}

    def single_eval(self, model_name):
        self.model = self.get_model()
        params = self.get_params()
        params['encoder'] = self.model

        se = senteval.engine.SE(params, self.batcher)
        transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark']
        results = se.eval(transfer_tasks)

        print_header = [model_name, 'pearson-r', 'peason-p_val', 'spearman-r', 'spearman-p_val', 'n_samples']
        print_contents = [print_header]

        print_all_header = [model_name, 'pearson-wmean', 'spearman-wmean', 'pearson-mean', 'spearman-mean']
        print_all_contents = [print_all_header]

        for task in results:
            if task == 'STSBenchmark':
                print_all_contents.append([f'{task}-all',
                                           f'{Decimal(str(results[task]["pearson"] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP) :.2f}',
                                           f'{Decimal(str(results[task]["spearman"] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP) :.2f}',
                                           '-',
                                           '-'])
            else:
                for category in results[task].keys():
                    if category == 'all':
                        print_all_contents.append([f'{task}-{category}',
                                                   f'{Decimal(str(results[task][category]["pearson"]["wmean"] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP) :.2f}',
                                                   f'{Decimal(str(results[task][category]["spearman"]["wmean"] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP) :.2f}',
                                                   f'{Decimal(str(results[task][category]["pearson"]["mean"] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP) :.2f}',
                                                   f'{Decimal(str(results[task][category]["spearman"]["mean"] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP) :.2f}'])
                    else:
                        if self.with_detailed_log:
                            print_contents.append([f'{task}-{category}',
                                                   f'{Decimal(str(results[task][category]["pearson"][0] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP) :.2f}',
                                                   f'{results[task][category]["pearson"][1]}',
                                                   f'{Decimal(str(results[task][category]["spearman"][0] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP):.2f}',
                                                   f'{results[task][category]["spearman"][1]}',
                                                   f'{results[task][category]["nsamples"]}'])

        if self.with_reset_output_file:
            if os.path.exists(f'../results/{self.output_file_name}'):
                os.remove(f'../results/{self.output_file_name}')

        texts = []
        with open(f'../results/{self.output_file_name}', 'a') as f:
            for print_all_content in print_all_contents:
                text = ' '.join(['{: >40}'] + ['{: >18}'] * (len(print_all_header) - 1)).format(*print_all_content)
                print(text, file=f)
                print(text)
                texts.append(text)

            if self.with_detailed_log:
                print('', file=f)

                for print_content in print_contents:
                    print(' '.join(['{: >40}'] + ['{: >25}'] * (len(print_header) - 2) + ['{: >10}']).format(
                        *print_content), file=f)

            print('', file=f)
            print('', file=f)

        if self.with_save_embeddings:
            with open(f'../models/sentence_embeddings_{model_name}.pkl', 'wb') as f:
                pickle.dump(self.embeddings[model_name], f)

        return {'text': texts, 'pearson': results[task]["pearson"], 'spearman': results[task]["spearman"]}

    def modify_batch_sentences_for_senteval(self, batch_words):
        padded_sequences, padding_masks = {}, {}

        for model_name in self.model_names:
            items = []
            if model_name == 'glove':
                items = self.model.source[model_name].get_word_embedding(batch_words)
                items = [torch.FloatTensor(item) for item in items]
            else:
                for words in batch_words:
                    words = [w for w in words if w != '�']
                    item = self.model.source[model_name].get_word_embedding(' '.join(words))
                    items.append(torch.as_tensor(item['embeddings'][0], dtype=torch.float, device=self.device))
            padded_sequences[model_name] = torch.nn.utils.rnn.pad_sequence(items, batch_first=True)

            max_sentence_length = max([len(words) for words in batch_words])
            padding_masks[model_name] = torch.as_tensor([[[False] * len(words) + [True] * (max_sentence_length - len(words)) ] for words in batch_words], dtype=torch.bool, device=self.device).squeeze(1)

        max_sentence_length = [padded_sequences[k].shape[1] for k in padded_sequences.keys()]
        return padded_sequences, padding_masks


def prepare(params, samples):
    return



