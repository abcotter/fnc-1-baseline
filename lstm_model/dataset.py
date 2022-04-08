from torch.utils.data import Dataset
import torch
import pandas as pd
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
import time
import json
import sys


""""

series_to_word_indeces = {pandas idx: [indeces]}
indeces_to_words


"""

# do i need to include EOS/SOS tags

def save_data(data):
    data.to_csv('processed_data.csv')
    headlines = data['Headline'].tolist()
    bodies = data['articleBody'].tolist()
    unique_words = data['unique_words'].tolist()


    with open(f"headlines.json", 'w') as f:
        json.dump(headlines, f)

    with open(f"bodies.json", 'w') as f:
        json.dump(bodies, f)

    with open(f"unique_words.json", 'w') as f:
        json.dump(unique_words, f)
    

class Stances(Dataset):
    def __init__(self, stances_data_path, body_data_path, test=False, load_data=True):
        start = time.time()
        if load_data:
            self.data = pd.read_csv('processed_data.csv')
             # = df.sample(frac=1).reset_index(drop=True)
            with open('headlines.json') as f:
                self.headlines = json.load(f)
            with open('bodies.json') as f:
                self.bodies = json.load(f)
            with open('unique_words.json') as f:
                self.unique_words = json.load(f)

        else:
            stances = pd.read_csv(stances_data_path)
            body = pd.read_csv(body_data_path)
            self.stemmer = PorterStemmer()
            data = stances.merge(body, on='Body ID')
            data = data.apply(self.pre_process_text, axis=1)
            self.unique_words = data['unique_words'].tolist()
            self.headlines = data['Headline'].tolist()
            self.bodies = data['articleBody'].tolist()
            self.data = data
            save_data(data)

        if test:
            with open('word_to_index.json') as f:
                self.words_indices = json.load(f)
            with open('index_to_word.json') as f:
                self.index_to_words = json.load(f)
            unk_index = len(self.words_indices)

        else:
            self.all_unique_words = set()
            for l in self.unique_words:
                for w in l:
                    self.all_unique_words.add(w)
            self.words_indices = {w: i+1 for i, w in enumerate(self.all_unique_words)}
            self.index_to_words = {i+1: w for i, w in enumerate(self.all_unique_words)}
            self.words_indices['<pad>'] = 0
            self.index_to_words[0] = '<pad>'
            unk_index = len(self.all_unique_words) + 1
            self.words_indices['UNK'] = unk_index
            self.index_to_words[unk_index] = "UNK"

            with open('word_to_index.json', 'w') as f:
                json.dump(self.words_indices, f)

            with open('index_to_word.json', 'w') as f:
                json.dump(self.index_to_words, f)

        self.data_indeces = {}
        for idx in range(0, len(self.data)):
            headline = self.headlines[idx]

            body = self.bodies[idx]

            self.data_indeces[idx] = [[self.words_indices.get(w, unk_index) for w in headline],
                                      [self.words_indices.get(w, unk_index) for w in body]]
        end = time.time()
        self.categories = {"unrelated": 0, "agree": 1, "disagree": 2, "discuss": 3}
        print(f"Took {end-start} to process data")

    def pre_process_text(self, row):
        cols = ['Headline', 'articleBody']
        all_words = []
        for col in cols:
            t = row[col]
            t = t.lower()
            stop_words_punctuation = set(stopwords.words('english') + list(string.punctuation))
            words = [w for w in word_tokenize(t) if w not in stop_words_punctuation]
            words = [self.stemmer.stem(w) for w in words]
            row[col] = words
            all_words += words
        row['unique_words'] = list(set(all_words))
        return row

    def __len__(self):
        return len(self.data_indeces)

    def __getitem__(self, idx):
        y = self.data['Stance'].values[idx]
        y = self.categories[y]
        return (
            torch.tensor(self.data_indeces[idx][0]),
            torch.tensor(self.data_indeces[idx][1][:100]),
            torch.tensor([y], dtype=torch.long))

