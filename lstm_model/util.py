from torch.nn.utils.rnn import pad_sequence
import torch
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def pad_collate(batch):
  (xx, yy, zz) = zip(*batch)
  x_lens = [len(x) for x in xx]
  y_lens = [len(y) for y in yy]

  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
  zz = torch.tensor([z for z in zz])

  return xx_pad, yy_pad, zz, x_lens, y_lens

def evaluate(data_loader, model, validation_size, loss_fn, classes):
    model = model.eval()
    validation_losses = []
    num_correct_predictions = 0


    correct_labels = []
    predicted_labels = []
    with torch.no_grad():
      for batch, (headline, body, labels, headline_lengths, body_lengths) in enumerate(data_loader):
          output = model(headline, body)
          preds = torch.argmax(output, dim=1)
          correct_labels.extend(labels)
          predicted_labels.extend(preds)
          loss = loss_fn(output, labels)
          num_correct_predictions += torch.sum(preds == labels)
          validation_losses.append(loss.item())

      predicted_labels = torch.stack(predicted_labels).cpu()
      correct_labels = torch.stack(correct_labels).cpu()
      print('predicted_labels', batch, predicted_labels)

    print(classification_report(correct_labels, predicted_labels, target_names=classes))
    return num_correct_predictions.item()/validation_size, np.mean(validation_losses)
