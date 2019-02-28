import torch
from torch.autograd import Variable
import torchvision.models as models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
import matplotlib.pyplot as plt

class ModifiedAlexNetModel(torch.nn.Module):
  def __init__(self):
    # print('ModifiedAlexNetModel init')
    super(ModifiedAlexNetModel, self).__init__()

    model = models.alexnet(pretrained=True)
    self.features = model.features
    self.classifier = model.classifier

    #*********************** CHOOSE LAYER TO RETRAIN *********************************** 
    # for param in self.features.parameters():
    #   param.requires_grad = False

    # *************** ImageNet **********************
    # AlexNet(
    #   (features): Sequential(
    #     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    #     (1): ReLU(inplace)
    #     (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    #     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    #     (4): ReLU(inplace)
    #     (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    #     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (7): ReLU(inplace)
    #     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (9): ReLU(inplace)
    #     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (11): ReLU(inplace)
    #     (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    #   )
    #   (classifier): Sequential(
    #     (0): Dropout(p=0.5)
    #     (1): Linear(in_features=9216, out_features=4096, bias=True)
    #     (2): ReLU(inplace)
    #     (3): Dropout(p=0.5)
    #     (4): Linear(in_features=4096, out_features=4096, bias=True)
    #     (5): ReLU(inplace)
    #     (6): Linear(in_features=4096, out_features=1000, bias=True)
    #   )

    self.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(9216, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 1000))
    # in_features = 4096
    # out_features = 1000
    # self.classifier[6] = nn.Linear(in_features, out_features)

  def forward(self, x):
    # print('ModifiedAlexNetModel forward')
    x = self.features(x)
    # print('Model Alexnet: ',x.shape)
    x = x.view(x.size(0), -1)
    # print('Model Alexnet: ',x.shape)
    x = self.classifier(x)
    # print('Model Alexnet: ',x.shape)
    return x

class FilterPrunner:
  def __init__(self, model):
    print('FilterPrunner init')
    self.model = model
    self.reset()
  
  def reset(self):
    print('FilterPrunner reset')

    # self.activations = []
    # self.gradients = []
    # self.grad_index = 0
    # self.activation_to_layer = {}
    self.filter_ranks = {}

  def forward(self, input):
    # print('FilterPrunner forward')


    self.activations = [] # convolutional layers list for ranking
    self.gradients = []
    self.grad_index = 0
    self.activation_to_layer = {}

    activation_index = 0
    for layer, (name, module) in enumerate(self.model.features._modules.items()):
        # print(' Layer, name, module: ', layer, name, module)
        input = module(input)
        if isinstance(module, torch.nn.modules.conv.Conv2d):
          input.register_hook(self.compute_rank) # applied compute rank on grad everytime backward called
          self.activations.append(input)
          self.activation_to_layer[activation_index] = layer
          activation_index += 1

    return self.model.classifier(input.view(input.size(0), -1))

  def compute_rank(self, grad):   
    # print('FilterPrunner compute rank')
    # print('*********************** START RANKING FILER')
    # print('************************************* GRADE SHAPE: ', grad.shape)
    # print(grad.shape)

    activation_index = len(self.activations) - self.grad_index - 1
    activation = self.activations[activation_index]
    # print('********************** Activation: ', activation)
    # print('********************** grad: ', grad)

    values =torch.sum((activation * grad), dim=0, keepdim = True).sum(dim=2, keepdim = True).sum(dim=3,keepdim = True)[0, :, 0, 0].data
         
    # Normalize the rank by the filter dimensions
    values = values / (activation.size(0) * activation.size(2) * activation.size(3))
      
    if activation_index not in self.filter_ranks:
      self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_().cuda()
        

    self.filter_ranks[activation_index] += values
    self.grad_index += 1

  def lowest_ranking_filters(self, num):
    print('FilterPrunner lowest ranking filters')

    data = []
    for i in sorted(self.filter_ranks.keys()):
      for j in range(self.filter_ranks[i].size(0)):
        data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

    return nsmallest(num, data, itemgetter(2))

  def normalize_ranks_per_layer(self):
    print('FilterPrunner normalize ranks per layer')
    for i in self.filter_ranks:
      v = torch.abs(self.filter_ranks[i])
      
      # v = v / np.sqrt(torch.sum(v * v))
      temp = torch.sum(v * v).cpu().numpy()
      v = v / np.sqrt(temp)
      self.filter_ranks[i] = v.cpu()

  def get_prunning_plan(self, num_filters_to_prune):
    print('FilterPrunner get_prunning plan')
    filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

    # After each of the k filters are prunned,
    # the filter index of the next filters change since the model is smaller.
    filters_to_prune_per_layer = {}
    for (l, f, _) in filters_to_prune:
      if l not in filters_to_prune_per_layer:
        filters_to_prune_per_layer[l] = []
      filters_to_prune_per_layer[l].append(f)

    for l in filters_to_prune_per_layer:
      filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
      for i in range(len(filters_to_prune_per_layer[l])):
        filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

    filters_to_prune = []
    for l in filters_to_prune_per_layer:
      for i in filters_to_prune_per_layer[l]:
        filters_to_prune.append((l, i))

    return filters_to_prune        

class PrunningFineTuner_AlexNet:
  def __init__(self, train_path, test_path, model):
    print(' PrunningFineTuner_AlexNet init')
    self.train_data_loader = dataset.loader(train_path)
    self.test_data_loader = dataset.test_loader(test_path)

    self.model = model
    self.criterion = torch.nn.CrossEntropyLoss()
    print('PrunningFineTuner_AlexNet init filter prunner')
    self.prunner = FilterPrunner(self.model)

    self.model.train()

  def accuracy(self,output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.numpy()[0])
            # print("***************** Accuracy top K:             ", res)
            # res.append(correct_k.mul_(100.0 / batch_size))
    return res

  def test(self):
    print(' PrunningFineTuner_AlexNet test')
    self.model.eval()
    correct = 0
    total = 0
    correct1 = 0
    correct5 = 0

    for i, (batch, label) in enumerate(self.test_data_loader):
      batch = batch.cuda()
      # print('PrunningFineTuner_AlexNet test calculate output')
      output = self.model(Variable(batch))
      # print('******************************* Output data: ',output.shape)
      corr1, corr5 = self.accuracy(output, label, topk=(1, 5))

      pred = output.data.max(1)[1]
      correct += pred.cpu().eq(label).sum()
      # print('******************* Correct: ', correct)
      total += label.size(0)
      correct1 += corr1
      correct5 += corr5
     
    accuracy = float(correct) / total
    self.model.train()
    print("Accuracy :", accuracy)
    acc1 = float(correct1) / total
    acc5 = float(correct5) / total 
    print("Accuracy top1 and top5: ", acc1, acc5)
    # return accuracy
    return acc1, acc5

  def train(self, optimizer = None, epoches = 10):
    print(' PrunningFineTuner_AlexNet train')
    accuracies1 = 0
    accuracies5 = 0
    if optimizer is None:
      optimizer = \
        optim.SGD(self.model.classifier.parameters(), 
          lr=0.0001, momentum=0.9)

    for i in range(epoches):
      print("Epoch: ", i)
      self.train_epoch(optimizer)
      acc1, acc5 = self.test()
      accuracies1 += acc1
      accuracies5 += acc5

    print("Finished fine tuning.")
    accuracy1 = accuracies1 / epoches
    accuracy5 = accuracies5 / epoches
    # return accuracy
    return accuracy1, accuracy5
  
  def train_epoch(self, optimizer = None, rank_filters = False):
    print(' PrunningFineTuner_AlexNet train epoch')
    batch_num = 0

    for batch, label in self.train_data_loader:
      if batch_num > 0:
       break
      print('************ batch number: ', batch_num)
      batch_num += 1
      self.train_batch(optimizer, batch.cuda(), label.cuda(), rank_filters)

  def train_batch(self, optimizer, batch, label, rank_filters):
    # print(' PrunningFineTuner_AlexNet train batch ')
    self.model.zero_grad()
    input = Variable(batch)

    if rank_filters:
      output = self.prunner.forward(input)
      self.criterion(output, Variable(label)).backward()
    else:
      self.criterion(self.model(input), Variable(label)).backward()
      optimizer.step()

  def get_candidates_to_prune(self, num_filters_to_prune):
    print(' PrunningFineTuner_AlexNet get candidates to prune')
    self.prunner.reset()

    self.train_epoch(rank_filters = True)
    
    self.prunner.normalize_ranks_per_layer()

    return self.prunner.get_prunning_plan(num_filters_to_prune)
    
  def total_num_filters(self):
    print(' PrunningFineTuner_AlexNet total number filters')
    filters = 0
    for name, module in self.model.features._modules.items():
      if isinstance(module, torch.nn.modules.conv.Conv2d):
        filters = filters + module.out_channels
    return filters

  # ***************************** PRUNE HERE **********************************
  def prune(self, model_name):
    print(' PrunningFineTuner_AlexNet prune')
    #Get the accuracy before prunning
    self.test()
    print(' ****************************** PrunningFineTuner_AlexNet prune before train')
    self.model.train()

    #Make sure all the layers are trainable
    for param in self.model.features.parameters():
      param.requires_grad = True

    number_of_filters = self.total_num_filters()
    print(' ****************************** PrunningFineTuner_AlexNet Total Number of filters: ', number_of_filters)
    # num_filters_to_prune_per_iteration = int(number_of_filters / 16)
    num_filters_to_prune_per_iteration = 1
    print(' ****************************** PrunningFineTuner_AlexNet prune Number of pruned filters each iteration: ', num_filters_to_prune_per_iteration)
    iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration) - 1
    epoch_num = 1

    # iterations = int(iterations * 2.0 / 3)   
    # print("Number of prunning iterations to reduce 67% filters", iterations)
    # iterations = 2  
    print("Number of prunning iterations ", iterations)

    pruned_percents = []
    pruned_accuracies1 = []
    pruned_accuracies5 = []
    finetuned_accuracies1 = []
    finetuned_accuracies5 = []
    # ***************************** RANKING FILTERS TO PRUNE **********************************
    for _ in range(iterations):
      print( " Ranking filters.. ")
      prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
      layers_prunned = {}
      for layer_index, filter_index in prune_targets:
        if layer_index not in layers_prunned:
          layers_prunned[layer_index] = 0
        layers_prunned[layer_index] = layers_prunned[layer_index] + 1 

      print ("************************* Layers that will be prune", layers_prunned)

      #*********************************** START PRUNING *********************************
      print ("Prunning filters.. ")
      model = self.model.cpu()
      for layer_index, filter_index in prune_targets:
        # print('Layer and filter index: ', layer_index, filter_index)
        model = prune_alexnet_conv_layer(model, layer_index, filter_index)

      self.model = model.cuda()
      print(' *************** Pruned: ', number_of_filters, self.total_num_filters())
      pruned_percent = 100 * (1 - float(self.total_num_filters()) / number_of_filters)
      pruned_percents.append(pruned_percent)
      message = str(pruned_percent) + "%"

      # message = str(100* number_of_filters / float(self.total_num_filters())) + "%"
      print( "Filters prunned", str(message))

      # ****************************** TEST ACCURACY AFTER PRUNING **********************************
      pruned_acc1, pruned_acc5 = self.test()
      pruned_accuracies1.append(pruned_acc1)
      pruned_accuracies5.append(pruned_acc5)
      # ********************************** RETRAIN AFTER PRUNED ****************************************
      print( "*********************************Fine tuning to recover from prunning iteration.")
      optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
      acc1, acc5 = self.train(optimizer, epoches = epoch_num)
      finetuned_accuracies1.append(acc1)
      finetuned_accuracies5.append(acc5)

      print('******************* Pruned accuracies top 5: ', pruned_accuracies5)
      print('******************* Finetuned accuracies top 5: ', finetuned_accuracies5)

    #**************** COMPARED GRAPH ***********************
    plt.title("Finetune vs Pruned Accuracy")
    plt.xlabel("Pruned percents")
    plt.ylabel("Validation Accuracy")
    plt.plot(pruned_percents, pruned_accuracies5, label='pruned')
    plt.plot(pruned_percents, finetuned_accuracies5, label='finetune')
    # plt.xlim((0,100))
    # plt.ylim((0,1.))
    plt.xticks(np.arange(0, 100, 10.0))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.legend()
    plt.show()
 
    # print( "Finished. Going to fine tune the model a bit more")
    # self.train(optimizer, epoches = epoch_num)
    pruned_model_name = model_name + "_prunned"
    torch.save(model.state_dict(), pruned_model_name)

def get_args(data_path):
    print('*********** Get args **************** ')

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")

    train_path =  data_path + "\\train"
    test_path = data_path + "\\val"

    train_path = data_path + "/train"
    test_path = data_path + "/val"

    parser.add_argument("--train_path", type = str, default = train_path)
    parser.add_argument("--test_path", type = str, default = test_path)
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()

    return args

def main():
  print('************************ Main function')

  data_dir = r"C:\Users\Thanh\code\repos\data"

  data_list = ['hymenoptera_data','dogs_cats', 'small_imagenet', 'imagenet'] 
  data_name = data_list[3]

  model_name = data_name + '_model'

  data_path = data_dir + '\\' + data_name

  data_neptune = "/soc_local/data/pytorch/imagenet"

  epoch_num = 10
  # args = get_args(data_path)
  args = get_args(data_neptune)

  if args.prune:
    print('************************ Main function load model for pruning')
    model = torch.load(model_name).cuda()
    fine_tuner = PrunningFineTuner_AlexNet(args.train_path, args.test_path, model)
    print('Main function prunning')

    begin = time.time()
    fine_tuner.prune(model_name)
    benchmark = time.time() - begin
    print('Pruning take: ', benchmark)

  else:
    print('************************ Main function training ')

    model = ModifiedAlexNetModel().cuda()
    fine_tuner = PrunningFineTuner_AlexNet(args.train_path, args.test_path, model)
    begin = time.time()
    fine_tuner.train(epoches = epoch_num)
    benchmark = time.time() - begin
    print('Training take: ', benchmark)

    torch.save(model, model_name)

if __name__ == '__main__':
  main()
