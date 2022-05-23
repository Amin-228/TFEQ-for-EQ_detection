import torch
import torch.nn.functional as F
import numpy as np
import random
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, roc_auc_score


def train(model, train_loader, optimizer, epoch):
    model.train()
    loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
#         loss += F.cross_entropy( scores, captions )
        l = F.nll_loss(output, target.squeeze())
        l.backward()
        optimizer.step()
        loss.append(l.item())
    loss = np.mean(loss)
    print('Train Epoch: {}, Average Loss: {:.4f}'.format(epoch, loss))
    return loss


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target.squeeze(),
                                    reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]  #
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def summary(target, y_pred):
    print("confusion matrix:\n", confusion_matrix(target, y_pred))
    print('Roc score: %.4f' % roc_auc_score(target, y_pred))
    print("F1 score: %.4f" % f1_score(target, y_pred))
    print(classification_report(target, y_pred))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
