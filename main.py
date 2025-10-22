#!/usr/bin/env python3
import os
import numpy
from numpy import random
import scipy
import matplotlib
import pickle
matplotlib.use('agg')
from matplotlib import pyplot

import torch
import torchvision
## you may wish to import other things like torch.nn

### hyperparameter settings and other constants
batch_size = 100
num_classes = 10
epochs = 10
mnist_input_shape = (28, 28, 1)
d1 = 1024
d2 = 256
alpha = 0.1
beta = 0.9
alpha_adam = 0.001
rho1 = 0.99
rho2 = 0.999
### end hyperparameter settings


# load the MNIST dataset using TensorFlow/Keras
def load_MNIST_dataset():
	train_dataset = torchvision.datasets.MNIST(
		root = './data',
		train = True,
		transform = torchvision.transforms.ToTensor(),
		download = True)
	test_dataset = torchvision.datasets.MNIST(
		root = './data',
		train = False,
		transform = torchvision.transforms.ToTensor(),
		download = False)
	return (train_dataset, test_dataset)

# construct dataloaders for the MNIST dataset
#
# train_dataset        input train dataset (output of load_MNIST_dataset)
# test_dataset         input test dataset (output of load_MNIST_dataset)
# batch_size           batch size for training
# shuffle_train        boolean: whether to shuffle the training dataset
#
# returns              tuple of (train_dataloader, test_dataloader)
#     each component of the tuple should be a torch.utils.data.DataLoader object
#     for the corresponding training set;
#     use the specified batch_size and shuffle_train values for the training DataLoader;
#     use a batch size of 100 and no shuffling for the test data loader
def construct_dataloaders(train_dataset, test_dataset, batch_size, shuffle_train=True):
	train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=shuffle_train)
	test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=100,shuffle=shuffle_train)
	return(train_dataloader,test_dataloader)


# evaluate a trained model on MNIST data
#
# dataloader    dataloader of examples to evaluate on
# model         trained PyTorch model
# loss_fn       loss function (e.g. torch.nn.CrossEntropyLoss)
#
# returns       tuple of (loss, accuracy), both python floats
@torch.no_grad()
def evaluate_model(dataloader, model, loss_fn):
	loss = 0
	num_correct = 0.0
	num_total = 0.0
	for (X,y) in dataloader:
		output = model(X)
		loss += loss_fn(output,y).item()
		for i in range(len(output)):
			num_total += 1
			if (output[i].argmax() == y[i]):
				num_correct += 1
	return(loss,num_correct/num_total)


# build a fully connected two-hidden-layer neural network for MNIST data, as in Part 1.1
# use the default initialization for the parameters provided in PyTorch
#
# returns   a new model of type torch.nn.Sequential
def make_fully_connected_model_part1_1():
	model = torch.nn.Sequential(
		torch.nn.Flatten(),
		torch.nn.Linear(in_features=784,out_features=1024),
		torch.nn.ReLU(),
		torch.nn.Linear(in_features=1024,out_features=256),
		torch.nn.ReLU(),
		torch.nn.Linear(in_features=256,out_features=10)
	)
	return model

# build a fully connected two-hidden-layer neural network with Batch Norm, as in Part 1.4
# use the default initialization for the parameters provided in PyTorch
#
# returns   a new model of type torch.nn.Sequential
def make_fully_connected_model_part1_4():
	model = torch.nn.Sequential(
		torch.nn.Flatten(),
		torch.nn.Linear(in_features=784, out_features=1024),
		torch.nn.BatchNorm1d(num_features=1024),
		torch.nn.ReLU(),
		torch.nn.Linear(in_features=1024, out_features=256),
		torch.nn.BatchNorm1d(num_features=256),
		torch.nn.ReLU(),
		torch.nn.Linear(in_features=256, out_features=10),
		torch.nn.BatchNorm1d(num_features=10)
	)
	return model
# build a convolutional neural network, as in Part 3.1
# use the default initialization for the parameters provided in PyTorch
#
# returns   a new model of type torch.nn.Sequential
def make_cnn_model_part3_1():
	model = torch.nn.Sequential(
		torch.nn.Conv2d(1,16,(3,3),1,0),
		torch.nn.BatchNorm2d(16),
		torch.nn.ReLU(),
		torch.nn.Conv2d(16,16,(3,3),1,0),
		torch.nn.BatchNorm2d(16),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d((2,2)),
		torch.nn.Conv2d(16,32,(3,3),1,0),
		torch.nn.BatchNorm2d(32),
		torch.nn.ReLU(),
		torch.nn.Conv2d(32,32,(3,3),1,0),
		torch.nn.BatchNorm2d(32),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d((2,2)),
		torch.nn.Flatten(),
		torch.nn.Linear(196,128),
		torch.nn.ReLU(),
		torch.nn.Linear(128,10)
	)
	return model 

# train a neural network on MNIST data
#     be sure to call model.train() before training and model.eval() before evaluating!
#
# train_dataloader   training dataloader
# test_dataloader    test dataloader
# model              dnn model to be trained (training should mutate this)
# loss_fn            loss function
# optimizer          an optimizer that inherits from torch.optim.Optimizer
# epochs             number of epochs to run
# eval_train_stats   boolean; whether to evaluate statistics on training set each epoch
# eval_test_stats    boolean; whether to evaluate statistics on test set each epoch
#
# returns   a tuple of
#   train_loss       an array of length `epochs` containing the training loss after each epoch, or [] if eval_train_stats == False
#   train_acc        an array of length `epochs` containing the training accuracy after each epoch, or [] if eval_train_stats == False
#   test_loss        an array of length `epochs` containing the test loss after each epoch, or [] if eval_test_stats == False
#   test_acc         an array of length `epochs` containing the test accuracy after each epoch, or [] if eval_test_stats == False
#   approx_tr_loss   an array of length `epochs` containing the average training loss of examples processed in this epoch
#   approx_tr_acc    an array of length `epochs` containing the average training accuracy of examples processed in this epoch
def train(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs, eval_train_stats=True, eval_test_stats=True):
	train_loss = []
	train_acc = []
	test_loss = []
	test_acc = []
	approx_tr_loss = []
	approx_tr_acc = []
	for e in range(epochs):
		model.train()
		epoch_loss = 0
		epoch_num_correct = 0.0
		epoch_num_total = 0.0
		for (X,y) in train_dataloader:
			output = model(X)
			loss = loss_fn(output,y)
			epoch_loss += loss.item()
			epoch_num_correct += (y == output.argmax(dim=1)).sum().item() 
			epoch_num_total += y.shape[0]
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		approx_tr_loss.append(epoch_loss)
		approx_tr_acc.append(epoch_num_correct/epoch_num_total)

		if (eval_train_stats):
			model.eval()
			my_loss,my_acc = evaluate_model(train_dataloader,model,loss_fn)
			train_loss.append(my_loss)
			train_acc.append(my_acc)
		if (eval_test_stats):
			model.eval()
			my_loss,my_acc = evaluate_model(test_dataloader,model,loss_fn)
			test_loss.append(my_loss)
			test_acc.append(my_acc)
	return (train_loss,train_acc,test_loss,test_acc,approx_tr_loss,approx_tr_acc)

""" Run your function to this network with a cross entropy loss (hint: you will find the torch.nn.CrossEntropyLoss loss from PyTorch to be useful here) using stochastic gradient descent (SGD). Use the following hyperparameter settings and instructions:
Learning rate α=0.1 . Minibatch size B=100
Run for 10 epochs.
Your code should save the statistics that are output by train in addition to the total wall-clock time used for training. 
You should expect to get about 98% test accuracy here. """
def run_1_1(train_dataset,test_dataset):
	loss_fn = torch.nn.CrossEntropyLoss()
	model = make_fully_connected_model_part1_1()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
	train_dataloader,test_dataloader = construct_dataloaders(train_dataset,test_dataset,100)
	stats = train(train_dataloader,test_dataloader,model,loss_fn,optimizer,epochs=10)
	with open("stats_1.1.pkl","wb") as f:
		pickle.dump(stats,f)
	print("test accuracy: " + str(stats[3][-1]))

def run_1_2(train_dataset,test_dataset):
	loss_fn = torch.nn.CrossEntropyLoss()
	model = make_fully_connected_model_part1_1()
	optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1, alpha=0.9)
	train_dataloader,test_dataloader = construct_dataloaders(train_dataset,test_dataset,100)
	stats = train(train_dataloader,test_dataloader,model,loss_fn,optimizer,epochs=10)
	with open("stats_1.2.pkl","wb") as f:
		pickle.dump(stats,f)
	print("test accuracy: " + str(stats[3][-1]))

def run_1_3(train_dataset,test_dataset):
	loss_fn = torch.nn.CrossEntropyLoss()
	model = make_fully_connected_model_part1_1()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.99,0.999))
	train_dataloader,test_dataloader = construct_dataloaders(train_dataset,test_dataset,100)
	stats = train(train_dataloader,test_dataloader,model,loss_fn,optimizer,epochs=10)
	with open("stats_1.3.pkl","wb") as f:
		pickle.dump(stats,f)
	print("test accuracy: " + str(stats[3][-1]))

def run_1_4(train_dataset,test_dataset):
	loss_fn = torch.nn.CrossEntropyLoss()
	model = make_fully_connected_model_part1_1()
	optimizer = torch.optim.RMSProp(model.parameters(), lr=0.001, alpha=0.9)
	train_dataloader,test_dataloader = construct_dataloaders(train_dataset,test_dataset,100)
	stats = train(train_dataloader,test_dataloader,model,loss_fn,optimizer,epochs=10)
	with open("stats_1.4.pkl","wb") as f:
		pickle.dump(stats,f)
	print("test accuracy: " + str(stats[3][-1]))


if __name__ == "__main__":
	(train_dataset, test_dataset) = load_MNIST_dataset()
	run_1_2(train_dataset,test_dataset)
	run_1_3(train_dataset,test_dataset)
