from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import torchvision.models as models
import argparse
import logging
import torchvision.transforms as transforms
import os
import torch.utils.data
import torch.cuda as cuda
import pickle
from math import exp
from torch.utils.data import Dataset, DataLoader

from util import mvcnn_VGG_avg, load_data, mvDataset


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-e', '--epochs', action='store', default=20, type=int, help='epochs (default: 40)')
parser.add_argument('--batchSize', action='store', default=8, type=int, help='batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', action='store', default=0.00001, type=float, help='learning rate (default: 0.00001)')
parser.add_argument('--m', '--momentum', action='store', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--train_f', action='store_false', default=True, help='Flag to train (STORE_FALSE)(default: True)')
parser.add_argument('--gpu_num','--list', type=int , nargs='+', help='gpu_num', required=True)
parser.add_argument('--trial', action='store', default=0, type=int, help='trial (default: 0)')
parser.add_argument("--net", default='VGG_avg', const='VGG_avg',nargs='?', choices=['VGG_avg'], help="net model(default:vgg16)")
parser.add_argument("--dataset", default='ModelNet', const='ModelNet',nargs='?', choices=['ModelNet','ObjectPI'], help="Dataset (default:ModelNet)")
arg = parser.parse_args()


def main():
    # create model directory to store/load old model
    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('log'):
        os.makedirs('log')

	# Logger Setting
    logger = logging.getLogger('netlog')
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler('log/logfile_'+arg.net+'_'+arg.dataset+ '_' + str(arg.trial)+ '.log')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # Batch size setting
    batch_size = arg.batchSize
    
    # load the data
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    #print(torch.cuda.device_count())
    if len(arg.gpu_num) == 1:
        torch.cuda.set_device(arg.gpu_num[0])

    # dataset directory
    if arg.dataset == 'ModelNet':
        pickle_filename = '../../modelnet.pickle'
        
        # Read list of training and validation data
        output_class = 40
        input_view = 12

    elif arg.dataset == 'ObjectPI':

        pickle_filename = '../../ObjectPI.pickle'
        
        # Read list of training and validation data
        output_class = 25
        input_view = 8  
     
        
    train, test = load_data(pickle_filename)  
    if arg.train_f:
        dataset_train = mvDataset(train, max_view=input_view, transform =data_transforms['train'])
        dataloader_trn = DataLoader(dataset_train, batch_size=arg.batchSize, shuffle=True, num_workers=4)
        dataset_val = mvDataset(test, max_view=input_view,  transform =data_transforms['val'])
        dataloader_val = DataLoader(dataset_val, batch_size=arg.batchSize, shuffle=True, num_workers=4)
        
    dataset_test = mvDataset(test, max_view=input_view,  transform =data_transforms['test'])
    dataloader_test = DataLoader(dataset_test, batch_size=arg.batchSize, shuffle=False, num_workers=4)
 
    logger.info("================================================")
    logger.info("Learning Rate: {}".format(arg.lr))
    logger.info("Classifier: "+arg.net)
    logger.info("Dataset: "+arg.dataset)
    logger.info("Nbr of Epochs: {}".format(arg.epochs))        
    logger.info("Output classes: {}".format(output_class))
    logger.info("Input view: {}".format(input_view))
    logger.info("================================================")

    
    if arg.net == 'VGG_avg':
        model = mvcnn_VGG_avg(input_view, output_class)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)
    # for gpu mode
    if len(arg.gpu_num) > 1:
        model = nn.DataParallel(model, arg.gpu_num)
    model.cuda()
    model_path = 'model/model_'+arg.net+'_'+arg.dataset+ '_' + str(arg.trial)+'.pt'

    # training
    print("Start Training")
    logger.info("Start Training")
    epochs = arg.epochs if arg.train_f else 0
    min_accuracy = 0
    

    correct, ave_loss = 0, 0
    # use cross entropy loss
    criterion = nn.CrossEntropyLoss()  
  
    for epoch in range(epochs):
        # training
        batch_idx = 0
        model.train()
        overall_acc = 0
        for batch_idx, (input, target) in enumerate(dataloader_trn):
            optimizer.zero_grad()
            input.requires_grad = True
            input, target = input.cuda(), target.cuda()
            outputs = model(input)
            loss = criterion(outputs, target)
            _, pred_label = torch.max(outputs.data, 1)
            correct = (pred_label.cpu().numpy() == target.cpu().detach().numpy()).sum()
            overall_acc += correct
            accuracy = 1.0*correct*1.0/batch_size
            loss.backward()              
            optimizer.step()             
            if batch_idx%50==0:
                print('==>>> epoch:{}, batch index: {}, train loss:{}, accuracy:{}'.format(epoch,batch_idx, loss.cpu().detach().numpy(), accuracy))
                logger.info('==>>> epoch:{}, batch index: {}, train loss:{}, accuracy:{}'.format(epoch,batch_idx, loss.cpu().detach().numpy(), accuracy))
            batch_idx += 1
            
            
        # Validation (always save the best model)
        print("Start Validation")
        logger.info("Start Validation")
        model.eval()
        # switch to evaluate mode
        correct, ave_loss = 0, 0
        for batch_idx, (input, target) in enumerate(dataloader_val):
            with torch.no_grad():
                input, target = input.cuda(), target.cuda()
                outputs = model(input)
                loss = criterion(outputs, target)
                _, pred_label = torch.max(outputs.data, 1)
                correct += (pred_label.cpu().numpy() == target.cpu().detach().numpy()).sum()
                ave_loss += loss.cpu().detach().numpy()
        accuracy = 1.0*correct*1.0/len(dataset_val)
        ave_loss /= len(dataset_val)
        if accuracy >= min_accuracy:
            min_accuracy = accuracy
            torch.save(model.state_dict(), model_path)
        print('==>>> test loss:{}, accuracy:{}'.format(ave_loss, accuracy))
        logger.info('==>>> test loss:{}, accuracy:{}'.format(ave_loss, accuracy))

        
    # Testing
    print("Start Testing for Best Model")
    logger.info("Start Testing for Best Model")
    if os.path.isfile(model_path):
        print("Loading model")
        logger.info("Loading model")
        model.load_state_dict(torch.load(model_path))
        
    # switch to evaluate mode
    model.eval()
    correct, ave_loss = 0, 0
    for batch_idx, (input, target) in enumerate(dataloader_test):
        with torch.no_grad():
            input, target = input.cuda(), target.cuda()
            criterion = nn.CrossEntropyLoss()
            outputs = model(input)
            loss = criterion(outputs, target)
            _, pred_label = torch.max(outputs.data, 1)
            correct += (pred_label.cpu().numpy() == target.cpu().detach().numpy()).sum()
            ave_loss += loss.cpu().detach().numpy()

    accuracy = 1.0*correct*1.0/len(dataset_test)
    ave_loss /= len(dataset_test)
    print('==>>> test loss:{}, accuracy:{}'.format(ave_loss, accuracy))
    logger.info('==>>> test loss:{}, accuracy:{}'.format(ave_loss, accuracy))
        
if __name__ == "__main__":
    main()