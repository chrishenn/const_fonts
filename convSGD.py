import os
import torch
import torch.hub
import torch.backends.cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.utils.data as data

import numpy as np
import time
import visdom

# Net1: 7-layer Convnet
class Net1(nn.Module):
    def __init__(self, num_classes):
        super(Net1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = nn.Linear(4704, 96)
        self.layer6 = nn.Linear(96, num_classes)
        self.layer7 = nn.Softmax(0)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)

        return out


class Net2(nn.Module):
    def __init__(self, num_classes):
        super(Net2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))


        self.layer3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer6 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))


        self.layer8 = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer9 = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))


        self.layer10 = nn.Linear(24, 24)
        self.layer11 = nn.Linear(24, 24)
        self.layer12 = nn.Linear(24, 24)
        self.layer13 = nn.Linear(24, num_classes)
        self.layer14 = nn.Softmax(0)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        out = out.reshape(out.size(0), -1)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)

        return out

def get_data(train_root, test_root, val_root, trans_type='normalize', rot_degree=0, size=(224,224)):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if trans_type == 'rotate':
        train_transform = transforms.Compose([
                                                transforms.RandomRotation(rot_degree, expand=True),
                                                transforms.Resize(size),

                                                transforms.ToTensor(),
                                                normalize               ])

        test_transform = transforms.Compose([
                                                transforms.RandomRotation(rot_degree, expand=True),
                                                transforms.Resize(size),

                                                transforms.ToTensor(),
                                                normalize               ])

        val_transform = transforms.Compose([
                                            transforms.RandomRotation(rot_degree, expand=True),
                                            transforms.Resize(size),

                                            transforms.ToTensor(),
                                            normalize               ])
    elif trans_type == 'normalize':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    else:
        print("Invalid trans_type parameter to get_data: exiting...\n")
        exit(1)


    train_dataset = datasets.ImageFolder(train_root, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_root, transform=test_transform)
    val_dataset = datasets.ImageFolder(val_root, transform=val_transform)

    return train_dataset, test_dataset, val_dataset


def get_loader(train_dataset, test_dataset, val_dataset, loader_type, batch_size, sample_weights=[]):
    num_workers = 6
    pin_memory = True

    train_size = len(train_dataset)
    test_size = len(test_dataset)
    val_size = len(val_dataset)

    if loader_type == 'sequential':
        train_sampler = data.SequentialSampler(train_dataset)
        test_sampler =  data.SequentialSampler(test_dataset)
        val_sampler =   data.SequentialSampler(val_dataset)

    elif loader_type == 'random':
        train_sampler = data.RandomSampler(train_dataset, replacement=False)
        test_sampler =  data.RandomSampler(test_dataset, replacement=False)
        val_sampler =   data.RandomSampler(val_dataset, replacement=False)

    elif loader_type == 'weighted_random':
        train_sampler = data.WeightedRandomSampler(sample_weights, train_size, replacement=False)
        test_sampler =  data.WeightedRandomSampler(sample_weights, test_size, replacement=False)
        val_sampler =   data.WeightedRandomSampler(sample_weights, val_size, replacement=False)

    else:
        print("invalid loader_type passed to get_loader: exiting...\n")
        exit(1)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(dataset=     train_dataset,
                                               batch_size=  batch_size,
                                               sampler=     train_sampler,
                                               num_workers= num_workers,
                                               pin_memory=  pin_memory)

    test_loader = torch.utils.data.DataLoader(dataset=      test_dataset,
                                              batch_size=   batch_size,
                                              sampler=      test_sampler,
                                              num_workers=  num_workers,
                                              pin_memory=   pin_memory)

    val_loader = torch.utils.data.DataLoader(dataset=       val_dataset,
                                              batch_size=   batch_size,
                                              sampler=      val_sampler,
                                              num_workers=  num_workers,
                                              pin_memory=   pin_memory)

    return train_loader, test_loader, val_loader



def train(model, optimizer, train_loader, test_loader, num_epochs, loss_func, device,
          class_list, log, model_type, vis=None, loss_win=None, log_interval=1, parallel=False):

    # lr_sched = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    update_count = 0
    loss = None

    for epoch in range(1, num_epochs+1):
        tic = time.time()

        model.train()
        pid = os.getpid()

        for batch_idx, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()
            output = model(data.to(device))

            loss = loss_func(output, target.to(device))

            if parallel: loss.mean().backward()
            else: loss.backward()

            optimizer.step()

            if (batch_idx+1) == len(train_loader):
                outstr = 'PID: {:<8} Epoch: {:<6}  [{:<5}/{:>5} ({:>3.0f}%)]  ({:>0.2f}s)    Loss: {:<6.4f}\n'.format(
                                                pid, epoch, epoch, num_epochs, 100.*epoch/num_epochs, time.time() - tic, loss.item())
                print(outstr)

                if vis is not None and (update_count+1)%log_interval == 0:
                    train_acc = test_epoch(model, device, train_loader, 'Train', loss_func, class_list, log)
                    test_acc = test_epoch(model, device, test_loader, 'Test', loss_func, class_list, log)

                    loss_arr =  np.array([loss.item()])
                    train_arr = np.array([100 - train_acc])
                    test_arr =  np.array([100 - test_acc])
                    common_X =  np.array([epoch])

                    vis.scatter( Y=loss_arr,  X=common_X, win=loss_win, update='append', name='Mean Epoch Loss')
                    vis.scatter( Y=train_arr, X=common_X, win=loss_win, update='append', name='Train Error %')
                    vis.scatter( Y=test_arr, X=common_X, win=loss_win, update='append', name='Test Error %')

                    log.writelines(outstr)
                    update_count=0

                else: update_count+=1

    save_modelfile(model.state_dict(), optimizer.state_dict(), loss, model_type)


def test_epoch(model, device, data_loader, data_str, loss_func, class_list, log):
    model.eval()
    test_loss = 0
    correct = 0

    num_classes =   len(class_list)
    class_correct = np.zeros(num_classes)
    class_total =   np.zeros(num_classes)
    data_size =     len(data_loader.dataset)

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))

            #sum loss from all batches into test_loss
            test_loss += loss_func(output, target.to(device), reduction='sum').item()

            # overall accuracy
            pred = output.max(1)[1]
            correct += pred.eq(target.to(device)).sum().item()

            #accuracy by category
            c = np.squeeze(pred == target.to(device))

            for i in range(len(target)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        for i in range(num_classes):
            if class_total[i] > 0:
                class_percent = 100 * class_correct[i] / class_total[i]
            else: class_percent = 0.0

            outstr = '{}: Top-1 accuracy on class {} : {:0.2f}/{:0.2f}, ({:0.2f}%)\n'.format(data_str, class_list[i], class_correct[i],
                                                                                     class_total[i], class_percent)
            print(outstr)
            log.writelines(outstr)

    test_loss /= len(data_loader.dataset)
    test_acc = 100 * correct / data_size

    outstr = '{}: Average loss: {:.4f}, Top-1 accuracy: {}/{} ({:.0f}%)\n'.format(data_str, test_loss, correct, data_size, test_acc)
    print(outstr)
    log.writelines(outstr)

    return test_acc



def get_device():
    if torch.cuda.is_available():

        device = torch.device('cuda:0')
        num_dev = torch.cuda.device_count()

        print("Found ", num_dev, " GPUs")
        print("device: ", device)
        return device, num_dev
    else:
        print("Pytorch reports cuda device not available: exiting")
        exit(1)



def init_vis(url_send):
    vis = visdom.Visdom(server=url_send, use_incoming_socket=False)

    loss_window = vis.scatter(  X=np.zeros((1,2)),
                                opts=dict(
                                    xlabel='Epoch',
                                    ylabel='Loss/Error',
                                    title='Loss and Error',
                                    legend=['Mean Epoch Loss', 'Train Error %', 'Test Error %']
                                          )
                                )
    return vis, loss_window



def get_model(model_type, num_classes, lr, momentum, freeze_depth=0, pre_trained=False):

    if model_type == 'resnet50':
        model = models.resnet50(pretrained=pre_trained)

        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(0)
        )

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum)

    elif model_type == 'resnext101_32x8d':

        model = torch.hub.load('pytorch/vision', 'resnext101_32x8d', pretrained=True)

        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(0)
        )

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum)

    elif model_type == 'resnet101':

        model = torch.hub.load('pytorch/vision', 'resnet101', pretrained=True)

        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(0)
        )

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum)

    elif model_type == 'resnext101_32x8d_pickle':
        model, optimizer = load_modelfile('resnext101_32x8d_pickle.pth', lr, momentum)

    elif model_type == 'resnet50_pickle':
        model, optimizer = load_modelfile('resnet50_pickle.pth', lr, momentum)

    else:
        model = Net2(num_classes)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum)


    return model, optimizer


def save_modelfile(model_state_dict, optimizer_state_dict, loss, model_type):
    print("saving model state for model ", model_type, " to file...")

    torch.save({
        'model_state_dict': model_state_dict,
        'loss': loss,
        'model_type': model_type
    }, model_type + '_pickle.pth')


def load_modelfile(path, lr, momentum):
    checkpoint = torch.load(path)

    print("loading model state for model ", checkpoint['model_type'], " from file...")

    model = eval('models.' + checkpoint['model_type'])()
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum)

    return model, optimizer


if __name__ == '__main__':

    ########### PARAMS ################################################################################################
    img_size = (224, 224)
    class_list = ['A', 'B', 'E', 'F', 'H', 'I', 'K', 'L', 'T', 'V', 'X', 'Y']


    train_root =        'images_shift_green_6'
    test_root =         'images_shift_blue_15'
    val_root =          'images_shift_blue_600'


    loader_type =       'random'
    trans_type =        'rotate'
    rot_degree =        360

    num_epochs =    11000
    num_classes =   12
    lr =            0.05
    batch_size =    50
    momentum =      0.9
    loss_func =     F.cross_entropy
    model_type =    'resnet50'
    pre_trained =   False
    freeze_depth =  0


    # To enable
    visualize =     False
    log_interval =  400
    url_send =      'localhost'
    # url_send =    '192.168.1.4'
    # default port: 8097

    # sample_weights = []  # for WeightedRandomSampler: len weights must be <= len total dataset loaded


    ########### INIT  #################################################################################################
    if visualize:
        vis, loss_win = init_vis(url_send)
    else: vis, loss_win = None, None

    torch.backends.cudnn.benchmark = True

    model, optimizer = get_model(model_type, num_classes, lr, momentum, freeze_depth, pre_trained)

    device, num_dev = get_device()
    if num_dev > 1: model = nn.DataParallel(model)

    model.to(device)

    train_dataset, test_dataset, val_dataset = get_data(train_root, test_root, val_root, trans_type, rot_degree, img_size)

    train_loader, test_loader, val_loader = get_loader(train_dataset, test_dataset, val_dataset, loader_type, batch_size)


    ########### TRAIN  ################################################################################################
    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    val_size = len(val_loader.dataset)

    log = open('log.txt', 'w+')
    logstr = list()
    logstr.append('TRAIN: ' + train_root + '\tTEST: ' + test_root + '\tVALIDATE: ' + val_root + '\n')
    logstr.append('TRAIN dataset size: {:<6}  TEST dataset size: {:<6}  VAL dataset size: {:<6}  Epochs train: {:<6}  ' \
                    'Angle rot: {:<6}  Sampler type: {:<12}  Model type: {:<8}\n'.format(
                        train_size, test_size, val_size, num_epochs, rot_degree, loader_type, model_type))
    for strg in logstr: print(strg)


    start = time.time()
    train(model, optimizer, train_loader, test_loader, num_epochs, loss_func, device, class_list,
                    log, model_type, vis, loss_win, log_interval, parallel=(num_dev > 1))

    logstr.append("Trained {:>4} epochs in {:0.2f}s\n".format(num_epochs, time.time()-start))
    for strg in logstr: print(strg)
    log.writelines(logstr)



    ########### VALIDATE  #############################################################################################
    str1 = "\nAccuracy on TRAIN dataset: \n--------------------------------------------\n"
    print(str1)
    log.writelines(str1)

    test_epoch(model, device, train_loader, 'Train', loss_func, class_list, log)


    str2 = "\nAccuracy on VAL dataset: \n--------------------------------------------\n"
    print(str2)
    log.writelines(str2)

    test_epoch(model, device, val_loader, 'Val', loss_func, class_list, log)
