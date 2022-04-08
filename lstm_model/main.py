from dataset import Stances
from model import StanceDetectionModel
from torch.utils.data import DataLoader
import time
from torch import nn, optim
from torch.utils.data import random_split
import torch
import numpy as np
from util import pad_collate, evaluate
from collections import defaultdict
import matplotlib.pyplot as plt
import sys

def train_unrelated_related(data_loader, model, training_size, loss_fn, optimizer):
    model.train()
    num_correct_predictions = 0
    losses = []
    model = model.train()
    for batch, (headline, body, labels, headline_lengths, body_lengths) in enumerate(data_loader):
        output = model(headline, body)
        preds = torch.argmax(output, dim=1)
        num_correct_predictions += torch.sum(preds == labels)
        loss = loss_fn(output, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    average_accuracy = num_correct_predictions.item()/training_size
    return average_accuracy, np.mean(losses)



if __name__ == "__main__":

    dataset = Stances('train_stances.csv', 'train_bodies.csv')
    model = StanceDetectionModel(dataset)
    classes = ['unrelated', 'agree', 'disagree', 'discuss']

    training_size = round(len(dataset)*0.8)
    validation_size = len(dataset)-training_size
    train_subset, validation_subset = random_split(dataset, [training_size, validation_size])
    train_loader = DataLoader(train_subset, batch_size=32, collate_fn=pad_collate)
    validation_loader = DataLoader(validation_subset, batch_size =32, collate_fn=pad_collate)

    epochs = 10
    best_accuracy = 0
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for i in range(epochs):
        print('Epoch #', i)
        train_accuracy, training_loss = train_unrelated_related(train_loader, model, training_size, loss_fn, optimizer)
        print('Validation Metrics')
        val_accuracy, validation_loss = evaluate(validation_loader, model, validation_size, loss_fn, classes)

        print({'Training Accuracy': train_accuracy,
            'Training Loss': training_loss})
        print({'Validation Accuracy': val_accuracy,
            'Validation Loss': validation_loss})

        if val_accuracy >= best_accuracy:
            torch.save(model.state_dict(), f'model_final')
            best_accuracy = val_accuracy
        else:
            print('Finished training')
            break


