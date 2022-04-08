from torch import nn
import numpy as np
import gensim
import gensim.downloader as api
import torch
import torch.nn.functional as F

EMBEDDING_DIM = 128

def create_embedding_matrix(word_indices):
    wv = api.load('word2vec-google-news-300')
    embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(word_indices)+1, EMBEDDING_DIM))
    for word, i in word_indices.items():
        try:
            embeddings_matrix[i] == wv[word]
        except KeyError:
            embeddings_vector = np.random.normal(scale=0.6, size=(EMBEDDING_DIM, ))
            embeddings_matrix[i] = embeddings_vector
    return embeddings_matrix

class StanceDetectionModel(nn.Module):
    def __init__(self, dataset):
        super(StanceDetectionModel, self).__init__()
        self.lstm_size = 128
        self.headline_lstm_size = 25
        self.num_classes = 4
        embedding_matrix = create_embedding_matrix(dataset.words_indices)
        num_embeddings, embeddings_dimension = embedding_matrix.shape
        self.embedding = nn.Embedding(num_embeddings, embeddings_dimension)
        self.embedding.load_state_dict({'weight': torch.tensor(embedding_matrix)})

        self.lstm_headline = nn.LSTM(input_size = embeddings_dimension,
                                     hidden_size = self.headline_lstm_size,
                                     num_layers = 1,
                                     batch_first=True,
                                     bidirectional=True)

        self.lstm_body = nn.LSTM(input_size = embeddings_dimension,
                                 hidden_size = self.lstm_size,
                                 num_layers = 1,
                                 batch_first=True,
                                 bidirectional=True)

        self.relu = nn.ReLU()

        #how to know how many elements in the softmax layer
        output_size = (2*self.lstm_size) + (2*self.headline_lstm_size)
        self.linear = nn.Linear(output_size, self.num_classes)


    def forward(self, headline, body):
        headline_embedding = self.embedding(headline)
        body_embedding = self.embedding(body)
        headline_output, (hidden_headline_state, cell_headline_state) = self.lstm_headline(headline_embedding)
        body_output, (hidden_body_state, cell_body_state) = self.lstm_body(body_embedding)
        headline_state = torch.cat((hidden_headline_state[-2,:,:], hidden_headline_state[-1,:,:]), dim = 1)
        body_state = torch.cat((hidden_body_state[-2,:,:], hidden_body_state[-1,:,:]), dim = 1)
        # headline_state = hidden_headline_state.squeeze(0)
        # body_state = hidden_body_state.squeeze(0)
        concatenated = torch.cat((headline_state, body_state), dim=1)
        relu_output = self.relu(concatenated)
        output = self.linear(relu_output)
        return F.softmax(output,dim=1)

