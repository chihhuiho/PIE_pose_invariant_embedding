from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import torchvision.models as models
from torchvision.models.vgg import model_urls as model_url_vgg
import argparse
import logging

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-e', '--epochs', action='store', default=40, type=int, help='epochs (default: 40)')
parser.add_argument('--batchSize', action='store', default=32, type=int, help='batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', action='store', default=0.0001, type=float, help='learning rate (default: 0.0001)')
parser.add_argument('--train_f', action='store_false', default=True, help='Flag to train (STORE_FALSE)(default: True)')
parser.add_argument('--gpu_num', action='store', default=0, type=int, help='gpu_num (default: 0)')
parser.add_argument("--net", default='vgg16', const='vgg16',nargs='?', choices=['vgg16'], help="net model(default:vgg16)")
parser.add_argument("--dataset", default='ModelNet', const='ModelNet',nargs='?', choices=['ModelNet', 'ObjectPI'], help="Dataset (default:ModelNet)")
parser.add_argument('--trial', action='store', default=1, type=int, help='trial (default: 1)')
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
    ch = logging.FileHandler('log/logfile_'+arg.net+'_'+arg.dataset+'_'+ str(arg.trial) + '.log')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Batch size setting
    batch_size = arg.batchSize
    
    # Set current device
    torch.cuda.set_device(arg.gpu_num)

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
	
    # dataset directory
    if arg.dataset == 'ObjectPI':
        train_path = '../../ObjectPI/train'
        val_path = '../../ObjectPI/test'
        test_path = '../../ObjectPI/test'
    elif arg.dataset == 'ModelNet':
        train_path = '../../modelnet40/train'
        val_path = '../../modelnet40/test'
        test_path = '../../modelnet40/test'
		    
    image_datasets = {}
    if arg.train_f:
        image_datasets['train'] = datasets.ImageFolder(os.path.join(train_path),data_transforms['train'])
        image_datasets['val'] = datasets.ImageFolder(os.path.join(val_path),data_transforms['val'])
    image_datasets['test'] = datasets.ImageFolder(os.path.join(test_path),data_transforms['test'])
        
    # use the pytorch data loader
    arr = ['train', 'val', 'test'] if arg.train_f else ['test']
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                  for x in arr}

    dataset_sizes = {x: len(image_datasets[x]) for x in arr}

    # get the number of class
    class_names = image_datasets['test'].classes
    
    logger.info("================================================")
    logger.info("Learning Rate: {}".format(arg.lr))
    logger.info("Classifier: "+arg.net)
    logger.info("Dataset: "+arg.dataset)
    logger.info("Epoch: {}".format(arg.epochs))
    logger.info("=========================================================")
    
    if arg.net == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier._modules['6']= nn.Linear(4096, out_features=len(class_names))
        
    # optimize all parameters
    for name,param in model.named_parameters():
        param.requires_grad=True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)
         
    model.cuda()    
    model_path = 'model/model_'+arg.net+'_'+arg.dataset + '_' + str(arg.trial) +'.pt'
    print(model_path)
    # training
    print("Start Training")
    logger.info("Start Training")

    epochs = arg.epochs if arg.train_f else 0
    min_accuracy = 0
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # trainning
        model.train()
        overall_acc = 0
        for batch_idx, (x, target) in enumerate(dataloaders['train']):
            optimizer.zero_grad()
            x.requires_grad = True  
            x, target = x.cuda(), target.cuda()       
            outputs = model(x)
            loss = criterion(outputs, target)
            _, pred_label = torch.max(outputs.data, 1)
            correct = (pred_label.cpu().numpy() == target.cpu().detach().numpy()).sum()
            overall_acc += correct
            accuracy = correct*1.0/batch_size
            loss.backward()              
            optimizer.step()             
            
            if batch_idx%100==0:
                print('==>>> epoch:{}, batch index: {}, train loss:{}, accuracy:{}'.format(epoch,batch_idx, loss.cpu().detach().numpy(), accuracy))
                logger.info('==>>> epoch:{}, batch index: {}, train loss:{}, accuracy:{}'.format(epoch,batch_idx, loss.cpu().detach().numpy(), accuracy))
            
            
        # Validation (always save the best model)
        print("Start Validation")
        logger.info("Start Validation")

        model.eval()
        correct, ave_loss = 0, 0
        for batch_idx, (x, target) in enumerate(dataloaders['val']):
            # for gpu mode
            with torch.no_grad():
                x, target = x.cuda(), target.cuda()
                outputs = model(x)
                loss = criterion(outputs, target)
                _, pred_label = torch.max(outputs.data, 1)
                correct += (pred_label.cpu().numpy() == target.cpu().detach().numpy()).sum()
                ave_loss += loss.cpu().detach().numpy()
            
        accuracy = correct*1.0/dataset_sizes['val']
        ave_loss /= dataset_sizes['val']
        # save the model if it is better than current one
        if accuracy >= min_accuracy:
            min_accuracy = accuracy
            torch.save(model.state_dict(), model_path)

        print('==>>> val loss:{}, accuracy:{}'.format(ave_loss, accuracy))
        logger.info('==>>> val loss:{}, accuracy:{}'.format(ave_loss, accuracy))
        
    
    
    # testing
    print("Start Testing")
    logger.info("Start Testing")
    if os.path.isfile(model_path):
        print("Loading model")
        logger.info("Loading model")
        model.load_state_dict(torch.load(model_path))
    else:
        print("No model")
        return
    
    model.eval()
    correct, ave_loss = 0, 0
    for batch_idx, (x, target) in enumerate(dataloaders['test']):
        with torch.no_grad():
            x, target = x.cuda(), target.cuda()   
            outputs = model(x)
            loss = criterion(outputs, target)
            _, pred_label = torch.max(outputs.data, 1)
            correct += (pred_label.cpu().numpy() == target.cpu().detach().numpy()).sum()
            ave_loss += loss.cpu().detach().numpy()
        
    accuracy = correct*1.0/dataset_sizes['test']
    ave_loss /= dataset_sizes['test']
    print('==>>> test loss:{}, accuracy:{}'.format(ave_loss, accuracy))
    logger.info('==>>> test loss:{}, accuracy:{}'.format(ave_loss, accuracy))
  
    
if __name__ == "__main__":
    main()
    
