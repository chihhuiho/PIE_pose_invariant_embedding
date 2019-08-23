import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
from torchvision.models.alexnet import model_urls
from torchvision.models.alexnet import model_urls as model_url_alexnet
from torchvision.models.vgg import model_urls as model_url_vgg
import torchvision.models as models
import pickle
from random import randint
import random
import numpy as np
import time
from PIL import Image


class Multiview_obj:
    def __init__(self, view_list, transform, num_of_view):
        # number of views
        self.V = num_of_view
        self.transform = transform
        # a list of views for a single object
        self.views = self._load_views(view_list, num_of_view)
       
    def _load_views(self, view_list, V):
        views = None

        sampled_view_list = np.random.choice(view_list, V, replace=False)
        for f in sampled_view_list:
            img = Image.open(f)
            if self.transform is not None:
                img = self.transform(img)
             
            img = img.unsqueeze(0)    
            if views is None:
                views = img
            else:
                views = torch.cat([views, img], 0)
        # views has shape [batchsize, # of view , 3, 224, 224]
        return views

class mvDataset:
    def __init__(self, data , max_view, transform = None, view_drop_out = False, sampled_view = None):
        self.data = data
        self.objs = data['objs']
        self.labels = data['labels']
        self.max_view = max_view
        self.transform = transform
        self.view_drop_out = view_drop_out
        self.sampled_view = sampled_view

    def __len__(self):
        return len(self.objs)

    def __shuffle__(self):
        z = zip(self.objs, self.labels)
        random.shuffle(z)
        self.objs, self.labels = [list(l) for l in zip(*z)]
        
    def __preprocess__(self, views_for_a_single_object, sampled_view):
        s = Multiview_obj(views_for_a_single_object, self.transform, sampled_view)           
        return s
    
    def __getitem__(self, idx):
        objs_batch = self.objs[idx]            
        labels_batch = self.labels[idx]
        if self.view_drop_out and self.sampled_view is None:
            self.sampled_view = randint(1, self.max_view)
        else: 
            self.sampled_view = self.max_view
        
        shape_object = self.__preprocess__(objs_batch, self.sampled_view) 
        x = shape_object.views
        y = labels_batch
        return (x,y)

def load_data(pickle_filename):

    train = {}
    train['objs'] = []
    train['labels'] = []
    test = {}
    test['objs'] = []
    test['labels'] = []
    with open(pickle_filename, "rb") as input_file:
        data = pickle.load(input_file)    
    
    for c in data['train']:
        for obj in data['train'][c]:
            train['objs'].append(data['train'][c][obj]['objs'])
            train['labels'].append(data['train'][c][obj]['labels'])
            
    for c in data['test']:
        for obj in data['test'][c]:
            test['objs'].append(data['test'][c][obj]['objs'])
            test['labels'].append(data['test'][c][obj]['labels'])
            
    return train, test
    
# use avg during the view pooling stage
class mvcnn_VGG_avg(nn.Module):
    def __init__(self, input_view, output_class):
        super(mvcnn_VGG_avg, self).__init__()        
        self.input_view = input_view
        VGG = models.vgg16(pretrained=True)
        self.features = VGG.features
        self.input_view = input_view
        self.output_class = output_class
        
        self.classifier1 = nn.Sequential(*list(VGG.classifier)[0:5])
        self.classifier2 = nn.Sequential(*list(VGG.classifier)[5:])
        self.classifier2._modules['1']= nn.Linear(4096, out_features=output_class)
        
    def forward(self, x, sampled_view):
        # x: (n, self.view, 3, 224, 224)        
        if sampled_view == 1:
            merge_result = None
            for view in range(self.input_view):
                view_feature = self.features(x[:,view]).view(x.shape[0], 25088)            
                view_feature = self.classifier1(view_feature).view(x.shape[0],1,4096)
                if merge_result is None:
                    merge_result = view_feature
                else:
                    merge_result = torch.cat([merge_result, view_feature],1)

            merge_result = merge_result.view(-1,4096)
            merge_result = self.classifier2(merge_result)
            return merge_result
        
        else:
            sampled_image = random.sample(range(self.input_view), sampled_view)
            merge_result = None
            for view in sampled_image:
                view_feature = self.features(x[:,view]).view(x.shape[0], 25088)            
                view_feature = self.classifier1(view_feature).view(x.shape[0],1,4096)
                if merge_result is None:
                    merge_result = view_feature
                else:
                    merge_result = torch.cat([merge_result, view_feature],1)

            merge_result = torch.mean(merge_result,1)
            merge_result = merge_result.view(merge_result.size(0), -1)        
            merge_result = self.classifier2(merge_result)

            return merge_result
