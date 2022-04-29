import sys
import random
from pathlib import Path
import torch
from decimal import Decimal, getcontext
getcontext().prec = 6

class STSBenchmarkDataset:
    def __init__(self, mode='train'):
        self.current = 0
        self.batch_size = 1

        self.mode = mode
        self.path = None
        if mode in ['train', 'dev', 'test']:
            self.path = Path(f'../data/sts-{mode}.csv')

        if self.path is None:
            sys.exit('Please set dataset type.')

        self.texts = None
        with self.path.open('r', encoding='utf-8') as f:
            self.texts = [self.get_data_dict(*line.strip().split('\t')) for line in f.readlines()]
        self.dataset_size = len(self.texts)
        self.batch_mode = 'full' # full, fixed

    @staticmethod
    def get_data_dict(genre, filename, year, index, score, sentence1, sentence2):
        return {'genre': genre,
                'filename': filename,
                'year': year,
                'index': int(index),
                'score': float(score),
                'sentence1': sentence1,
                'sentence2': sentence2}

    def shuffle(self):
        random.shuffle(self.texts)

    def get_batch(self):
        if self.current + self.batch_size < len(self.texts):
            batch = self.texts[self.current: self.current + self.batch_size]
            self.current += self.batch_size
        else:
            batch = self.texts[self.current:]
            self.current = len(self.texts)

        sentences1 = [b['sentence1'] for b in batch]
        sentences2 = [b['sentence2'] for b in batch]
        scores = [b['score']/5.0 for b in batch]
        return sentences1, sentences2, scores

    def reset(self, with_shuffle=False):
        self.current = 0
        if with_shuffle:
            self.shuffle()

    def is_batch_end(self, with_test_mode=False):
        if with_test_mode and self.current >= 100:
            return True
        if self.batch_mode == 'full':
            if self.current == self.dataset_size:
                return True
        elif self.batch_mode == 'fixed':
            if self.current >= int(self.dataset_size / 10 + 0.5):
                return True
        return False

    def __str__(self):
        return f'{self.current} / {self.dataset_size}'

class STSDataset(STSBenchmarkDataset):
    def __init__(self, mode='STS12'):
        self.current = 0
        self.batch_size = 128

        self.mode = mode
        self.path = Path(f'../data/downstream/STS/{mode}-en-test')

        if '12' in mode:
            self.tags = ['MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.OnWN', 'surprise.SMTnews']
            # STS.input.MSRpar.txt
            # STS.input.MSRvid.txt
            # STS.input.SMTeuroparl.txt
            # STS.input.surprise.OnWN.txt
            # STS.input.surprise.SMTnews.txt
        elif '13' in mode:
            self.tags = ['FNWN', 'headlines', 'OnWN']
            # STS.input.FNWN.txt
            # STS.input.headlines.txt
            # STS.input.OnWN.txt
        elif '14' in mode:
            self.tags = ['deft-forum', 'deft-news', 'headlines', 'images', 'OnWN', 'tweet-news']
            # STS.input.deft-forum.txt
            # STS.input.deft-news.txt
            # STS.input.headlines.txt
            # STS.input.images.txt
            # STS.input.OnWN.txt
            # STS.input.tweet-news.txt
        elif '15' in mode:
            self.tags = ['answers-forums', 'answers-students', 'belief', 'headlines', 'images']
            # STS.input.answers-forums.txt
            # STS.input.answers-students.txt
            # STS.input.belief.txt
            # STS.input.headlines.txt
            # STS.input.images.txt
        elif '16' in mode:
            self.tags = ['answer-answer', 'headlines', 'plagiarism', 'postediting', 'question-question']
            # STS.input.answer-answer.txt
            # STS.input.headlines.txt
            # STS.input.plagiarism.txt
            # STS.input.postediting.txt
            # STS.input.question-question.txt
        self.texts = []
        for tag in self.tags:
            with (self.path / f'STS.input.{tag}.txt').open('r') as f:
                sentences = f.readlines()
            with (self.path / f'STS.gs.{tag}.txt').open('r') as f:
                golds = f.readlines()

            for ss, gs in zip(sentences, golds):
                sentence1, sentence2 = ss.strip().split('\t')
                try:
                    if gs.strip() == '':
                        continue
                    else:
                        gold_score = float(Decimal(gs.strip())/Decimal(5.0))
                        self.texts.append(
                            {'sentence1': sentence1, 'sentence2': sentence2, 'score': gold_score, 'tag': tag})
                except:
                    print("")

        self.dataset_size = len(self.texts)

    def get_batch(self):
        if self.current + self.batch_size < len(self.texts):
            batch = self.texts[self.current: self.current + self.batch_size]
            self.current += self.batch_size
        else:
            batch = self.texts[self.current:]
            self.current = len(self.texts)

        sentences1 = [b['sentence1'] for b in batch]
        sentences2 = [b['sentence2'] for b in batch]
        scores = [b['score'] for b in batch]
        tags = [b['tag'] for b in batch]
        return sentences1, sentences2, scores, tags

    def is_batch_end(self, with_test_mode=False):
        if with_test_mode and self.current >= 100:
            return True
        if self.current >= self.dataset_size:
            return True
        return False

    def __str__(self):
        return f'{self.current} / {self.dataset_size}'




if __name__ == '__main__':
    s = STSDataset('STS15')
    while not s.is_batch_end():
        print(s.get_batch())


