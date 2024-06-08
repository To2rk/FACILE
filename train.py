import torch
import itertools
import time
import numpy as np
from tqdm import tqdm
from models import FACILE
from utils.utils import get_parameter_number, chooseDataset, setup_seed
from utils.utils import MalwareImageDataset, chooseDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import argparse


criterion = torch.nn.CrossEntropyLoss()

def test(model, test_loader):
    model.eval()
    test_loss = 0
    
    # all labels
    out_true = []
    out_pred = []

    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()

    with torch.no_grad():
        for _,(x, y)in tqdm(enumerate(test_loader),total =len(test_loader), leave = True):
            x, y = x.cuda(), y.cuda()

            y_pred = model(x)
            test_loss += criterion(y_pred, y).item() * x.size(0) 

            y_pred = torch.argmax(y_pred, dim=1)

            out_true += list(y.detach().cpu().numpy())
            out_pred += list(y_pred.detach().cpu().numpy())

            correct += (y_pred == y).sum().float()
            total += len(y)

        test_loss /= total
        test_acc = correct / total

        macro_pr = metrics.precision_score(out_true, out_pred, average='macro')
        macro_re = metrics.recall_score(out_true, out_pred, average='macro')
        macro_f1 = metrics.f1_score(out_true, out_pred, average='macro')

        return test_loss, test_acc, macro_pr, macro_re, macro_f1


def train(fold, model, train_loader, test_loader, args):

    t0 = time.time()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    lr_decay = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr * args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        ti = time.time()
        training_loss = 0.0
        train_acc = 0
        correct = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()

        for _,(x, y)in tqdm(enumerate(train_loader),total =len(train_loader), leave = True):
            x, y = x.cuda(), y.cuda()

            y_pred = model(x)

            loss = criterion(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)

            correct += (y_pred == y).sum().float()
            total += len(y)

            training_loss += loss.item() * x.size(0) 

            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 

            lr_decay.step()

        train_loss = training_loss / total
        train_acc = correct / total

        test_loss, test_acc, macro_pr, macro_re, macro_f1 = test(model, test_loader)

        print('\033[32mEpoch: {:03d}\033[0m'.format(epoch),
            'train_loss: {:.5f}'.format(train_loss),
            'train_acc: {:.5f}'.format(train_acc),
            'test_loss: {:.5f}'.format(test_loss),
            '\033[32mtest_acc: {:.5f}\033[0m'.format(test_acc),
            '\033[32mmacro_pr: {:.5f}\033[0m'.format(macro_pr),
            '\033[32mmacro_re: {:.5f}\033[0m'.format(macro_re),
            '\033[32mmacro_f1: {:.5f}\033[0m'.format(macro_f1),
            '\033[32mtime: {:0.2f}s\033[0m'.format(time.time() - ti))

    dateStr = time.strftime("%d-%b-%Y-%H:%M:%S", time.localtime())

    model_dir = args.models_dir + args.dataset + '/all-resized-32/' + dateStr + '-' + str(fold) + '.pt'
    torch.save(model.state_dict(), model_dir)
    print('Trained model saved to %s.pt' % dateStr)

    print("Total time = %ds" % (time.time() - t0))

    return test_acc, train_loss, test_loss, macro_pr, macro_re, macro_f1


def main(args):

    setup_seed(args.seed)

    dataset_dir = args.data_dir + args.dataset + '/all-resized-32/'
    dataset = MalwareImageDataset(dataset_dir, args.dataset)

    all_labels = []
    for _, (_,y) in enumerate(dataset):
        all_labels.append(y)

    num_classes = len(chooseDataset(args.dataset))

    # index
    test_acc = []
    test_loss = []
    train_loss = []
    macro_pr = []
    macro_re = []
    macro_f1 = []

    skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    skfold_subsets = []

    for _, (_, test_y) in enumerate(skfold.split(dataset,all_labels)):
        skfold_subsets.append(test_y)
    
    skfold_subsets_all = list(itertools.chain.from_iterable(skfold_subsets))

    for fold in range(10):

        train_ids = []
        # test_ids = []

        for i in range(int(args.trainingSetProp * 10)):
            train_ids.extend(skfold_subsets[fold+i if fold+i<10-1 else fold+i-10])

        test_ids = list(set(skfold_subsets_all).difference(set(train_ids)))

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        test_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_subsampler)

        model = FACILE(input_size=[3, 32, 32], classes=num_classes, args=args)
        model.cuda()
        total_num, trainable_num = get_parameter_number(model)
        print("'Total': {}, 'Trainable': {}".format(total_num, trainable_num))

        final_test_acc, final_train_loss, final_test_loss, final_macro_pr, final_macro_re, final_macro_f1 = train(fold, model, train_dataloader, test_dataloader, args)
        train_loss.append(final_train_loss.item())
        test_loss.append(final_test_loss.item())
        test_acc.append(final_test_acc.item())
        macro_pr.append(final_macro_pr)
        macro_re.append(final_macro_re)
        macro_f1.append(final_macro_f1)

    # average_train_loss = round(np.mean(train_loss), 5)   
    # average_test_loss = round(np.mean(test_loss), 5)   
    average_test_acc = round(np.mean(test_acc), 5)   
    average_macro_pr = round(np.mean(macro_pr), 5)   
    average_macro_re = round(np.mean(macro_re), 5)   
    average_macro_f1 = round(np.mean(macro_f1), 5)   

    print('average_test_acc: ', average_test_acc)
    print('average_macro_pr:', average_macro_pr)
    print('average_macro_re:', average_macro_re)
    print('average_macro_f1:', average_macro_f1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=100, help='Seed value')
    parser.add_argument('--dataset', type=str, default='Virus_MNIST', choices=['MalImg', 'BIG2015', 'Virus_MNIST'], help='Dataset name')
    parser.add_argument('--trainingSetProp', type=float, default=0.7, help='Proportion of training set')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Directory for data')
    parser.add_argument('--models_dir', type=str, default='./models/', help='Directory for models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.15, help='Learning rate')
    parser.add_argument('--args_k_0', type=int, default=8, help='Value for args_k_0')
    parser.add_argument('--args_k_1', type=int, default=3, help='Value for args_k_1')
    parser.add_argument('--args_k_2', type=int, default=6, help='Value for args_k_2')
    parser.add_argument('--rate', type=float, default=0.15, help='Rate value')
    parser.add_argument('--max_lr', type=float, default=1.5, help='Maximum learning rate')

    args = parser.parse_args()

    main(args=args)
    