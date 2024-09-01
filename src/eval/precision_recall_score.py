import torch
import torchvision

from classifier import Net
from sklearn.metrics import f1_score, multilabel_confusion_matrix, precision_score, recall_score

from src.utils.helpers import load_model

# init classifier
classifier_path = 'classifier.pth'
classifier_state_dict = torch.load(classifier_path, map_location=torch.device('cpu'))
classifier_model = Net()
classifier_model.eval()
classifier_model.load_state_dict(classifier_state_dict)


# load model
model = load_model('bitnet_mnist')
# model = load_model('baseline_mnist')

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=1000, shuffle=True)

y_pred_1 = []
y_pred_2 = []
y_true = []

with torch.no_grad():
    for data, target in test_loader:
        # prediction on the test_data
        output1 = classifier_model(data)
        pred1 = output1.data.max(1, keepdim=True)[1]  # get index/label of the max element in the second dim
        y_pred_1 += pred1

        # true labels
        y_true += target.data.view_as(pred1)

        # prediction on the reconstructed data
        x = data.view(1000, 784).to(model.device)
        recon_x, mu, logvar = model(x)
        recon_x = recon_x.detach().cpu().reshape(1000, 1, 28, 28)
        output2 = classifier_model(recon_x)
        pred2 = output2.data.max(1, keepdim=True)[1]
        y_pred_2 += pred2

print("precision of the classifier ", precision_score(y_true, y_pred_1, average='micro'))

print("precision score:", precision_score(y_true, y_pred_2, average='micro'))
print("recall score:", recall_score(y_true, y_pred_2, average='micro'))
print(f1_score(y_pred_1, y_pred_2, average=None))
print(f1_score(y_pred_1, y_pred_2, average='micro'))
print(f1_score(y_pred_1, y_pred_2, average='macro'))
print(f1_score(y_pred_1, y_pred_2, average='weighted'))
print(multilabel_confusion_matrix(y_pred_1, y_pred_2))
