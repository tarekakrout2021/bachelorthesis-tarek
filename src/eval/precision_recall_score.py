import sys

import numpy as np
import torch
import torchvision

from classifier import Net
from src.utils.helpers import load_model


def init_classifier() -> Net:
    classifier_path = 'classifier.pth'
    classifier_state_dict = torch.load(classifier_path, map_location=torch.device('cpu'))
    classifier_model = Net()
    classifier_model.eval()
    classifier_model.load_state_dict(classifier_state_dict)
    return classifier_model


def main(*args):
    # init classifier
    classifier_model = init_classifier()

    # load model
    model = load_model('bitnet_mnist')
    if args and args[0] == 'baseline':
        model = load_model('baseline_mnist')

    # load test data
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=1000, shuffle=True)

    pred_original, pred_recon, true_labels = [], [], []

    with torch.no_grad():
        for data, target in test_loader:
            # prediction on the test_data
            output1 = classifier_model(data)  # output1 shape (btach_size, 10)
            pred1 = output1.data.max(1, keepdim=True)[1].reshape(
                1000)  # get index/label of the max element in the second dim
            pred_original.extend(np.round(pred1))

            # true labels
            true_labels.extend(target.data.view_as(pred1))

            # prediction on the reconstructed data
            x = data.view(1000, 784).to(model.device)  # batch_size is 1000 and each image is 28x28
            recon_x, mu, logvar = model(x)
            recon_x = recon_x.detach().cpu().reshape(1000, 1, 28, 28)
            output2 = classifier_model(recon_x)
            pred2 = output2.data.max(1, keepdim=True)[1].reshape(1000)
            pred_recon.extend(np.round(pred2))

    pred_original = np.array(pred_original)
    pred_recon = np.array(pred_recon)
    print("precision of the classifier ", np.mean([pred_original == true_labels]))

    print("precision score:", np.mean([pred_original == pred_recon]))


if __name__ == '__main__':
    main(*sys.argv[1:])
