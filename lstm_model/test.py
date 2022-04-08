from dataset import Stances
from torch.utils.data import DataLoader
from util import pad_collate, evaluate
from model import RelatedUnrelatedModel
import torch
from sklearn.metrics import confusion_matrix, classification_report
model_one_mapping =["unrelated", "related"]
correct_mappings = ["unrelated", "related"]
def test_model(test_data, unrelated_related_model):
    unrelated_related_model.eval()
    correct_labels = []
    predicted_labels = []
    num_correct_predictions = 0

    for i, (headline, body, labels, headline_lengths, body_lenths) in enumerate(test_data):
        output_1 = unrelated_related_model(headline, body)
        pred = torch.argmax(output_1, dim=1)
        correct_labels.extend(labels)
        predicted_labels.extend(pred)
        num_correct_predictions += torch.sum(pred == labels)
    #     else:
    #         output_2 = agree_disagree_model(headline, body)
    #         pred = torch.argmax(output_2, dim=1)
    #         pred_label = model_two_mapping[pred.item()]
    #         print('pred 2', pred, pred_label)
    #         predicted_labels.append(pred_label)
    # print('predictions', predicted_labels)
    # print('real', correct_labels)
    print(classification_report(correct_labels, predicted_labels, target_names=['unrelated', 'related']))
    print(num_correct_predictions.item()/len(test_data))

if __name__ == "__main__":
    dataset = Stances('competition_test_stances_filtered.csv', 'competition_test_bodies.csv', test=True)
    test_loader = DataLoader(dataset, batch_size=100, collate_fn=pad_collate)
    embedding_sizes = ['25', '50', '100']
    # lstm_types = ['unidirectional', 'bi_directional']
    for size in embedding_sizes:
        print(f'Output for size {size}')
        model = RelatedUnrelatedModel(dataset)
        model.load_state_dict(torch.load(f'./state_dictunrelated_related'))
        model.eval()
        test_model(test_loader, model)
