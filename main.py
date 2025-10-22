#!/usr/bin/env python3
import os
import numpy
from numpy import random
import scipy
import matplotlib
import pickle
matplotlib.use('agg')
from matplotlib import pyplot
import time
import math
import torch
import torchvision
## you may wish to import other things like torch.nn

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def _sync_and_retrieve_time():
	if DEVICE.type == "cuda":
		torch.cuda.synchronize()
	return time.time()

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
	test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=100,shuffle=False)
	return(train_dataloader,test_dataloader)

# evaluate a trained model on MNIST data
#
# dataloader    dataloader of examples to evaluate on
# model         trained PyTorch model
# loss_fn       loss function (e.g. torch.nn.CrossEntropyLoss)
#
# returns       tuple of (loss, accuracy), both python floats
"""@torch.no_grad()
def evaluate_model(dataloader, model, loss_fn):
	loss = 0
	num_correct = 0.0
	num_total = 0.0
	model = model.to(DEVICE)
	loss_fn = loss_fn.to(DEVICE)
	for (X,y) in dataloader:
		X = X.to(DEVICE)
		y = y.to(DEVICE)
		output = model(X)
		loss += loss_fn(output,y).item() * batch_size
		for i in range(len(output)):
			num_total += 1
			if (output[i].argmax() == y[i]):
				num_correct += 1
	return(loss / num_total,num_correct/num_total)"""
@torch.no_grad()
def evaluate_model(dataloader, model, loss_fn):
    model = model.to(DEVICE)
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for X, y in dataloader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        out = model(X)
        bs = y.size(0)
        # loss_fn usually returns mean; scale by batch size to accumulate a sum
        loss_sum += loss_fn(out, y).item() * bs
        correct  += (out.argmax(dim=1) == y).sum().item()
        total    += bs
    return loss_sum / total, correct / total



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

def make_fully_connected_model_custom_part2_2(hidden_dim=512, num_hidden_layers=2):
	layers = [torch.nn.Flatten()]
	in_dim = 28 * 28
	for _ in range(num_hidden_layers):
		layers.append(torch.nn.Linear(in_features=in_dim, out_features=hidden_dim))
		layers.append(torch.nn.ReLU())
		in_dim = hidden_dim
	layers.append(torch.nn.Linear(in_features=in_dim, out_features=10))
	return torch.nn.Sequential(*layers)

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
		torch.nn.Linear(512,128),
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
	epoch_times = []
	model = model.to(DEVICE)
	loss_fn = loss_fn.to(DEVICE)
	for e in range(epochs):
		epoch_start = _sync_and_retrieve_time()
		model.train()
		epoch_loss = 0
		epoch_num_correct = 0.0
		epoch_num_total = 0.0
		for (X,y) in train_dataloader:
			X = X.to(DEVICE)
			y = y.to(DEVICE)
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
		epoch_times.append(_sync_and_retrieve_time() - epoch_start)
	return (train_loss,train_acc,test_loss,test_acc,approx_tr_loss,approx_tr_acc,epoch_times)

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
	start_time = _sync_and_retrieve_time()
	stats = train(train_dataloader,test_dataloader,model,loss_fn,optimizer,epochs=10)
	wall_time = _sync_and_retrieve_time() - start_time
	with open("stats_1.1.pkl","wb") as f:
		pickle.dump(stats,f)
	print("test accuracy: " + str(stats[3][-1]))
	print(f"training wall time: {wall_time}s")

def run_1_2(train_dataset,test_dataset):
	loss_fn = torch.nn.CrossEntropyLoss()
	model = make_fully_connected_model_part1_1()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
	train_dataloader,test_dataloader = construct_dataloaders(train_dataset,test_dataset,100)
	start_time = _sync_and_retrieve_time()
	stats = train(train_dataloader,test_dataloader,model,loss_fn,optimizer,epochs=10)
	wall_time = _sync_and_retrieve_time() - start_time
	with open("stats_1.2.pkl","wb") as f:
		pickle.dump(stats,f)
	print("test accuracy: " + str(stats[3][-1]))
	print(f"training wall time: {wall_time}s")

def run_1_3(train_dataset,test_dataset):
	loss_fn = torch.nn.CrossEntropyLoss()
	model = make_fully_connected_model_part1_1()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.99,0.999))
	train_dataloader,test_dataloader = construct_dataloaders(train_dataset,test_dataset,100)
	start_time = _sync_and_retrieve_time()
	stats = train(train_dataloader,test_dataloader,model,loss_fn,optimizer,epochs=10)
	wall_time = _sync_and_retrieve_time() - start_time
	with open("stats_1.3.pkl","wb") as f:
		pickle.dump(stats,f)
	print("test accuracy: " + str(stats[3][-1]))
	print(f"training wall time: {wall_time}s")

def run_1_4(train_dataset,test_dataset):
	loss_fn = torch.nn.CrossEntropyLoss()
	model = make_fully_connected_model_part1_4()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
	train_dataloader,test_dataloader = construct_dataloaders(train_dataset,test_dataset,100)
	start_time = _sync_and_retrieve_time()
	stats = train(train_dataloader,test_dataloader,model,loss_fn,optimizer,epochs=10)
	wall_time = _sync_and_retrieve_time() - start_time
	with open("stats_1.4.pkl","wb") as f:
		pickle.dump(stats,f)
	print("test accuracy: " + str(stats[3][-1]))
	print(f"training wall time: {wall_time}s")

def run_momentum_sgd_grid_search(train_dataset, test_dataset):
	loss_fn = torch.nn.CrossEntropyLoss()
	step_sizes = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
	best_lr = None
	best_acc = float("-inf")
	for lr in step_sizes:
		print(f"Running momentum SGD with lr={lr}")
		model = make_fully_connected_model_part1_1()
		optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
		train_dataloader, test_dataloader = construct_dataloaders(train_dataset, test_dataset, 100)
		start_time = _sync_and_retrieve_time()
		stats = train(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs=10,eval_test_stats=False,eval_train_stats=False)
		wall_time = _sync_and_retrieve_time() - start_time
		test_loss,test_accuracy = evaluate_model(test_dataloader,model,loss_fn)
		with open(f"stats_grid_momentum_sgd_lr_{lr}.pkl", "wb") as f:
			pickle.dump(stats, f)
		print("test accuracy: " + str(test_accuracy))
		print(f"training wall time: {wall_time}s")
		if test_accuracy is not None and test_accuracy > best_acc:
			best_acc = test_accuracy
			best_lr = lr
	print(f"Best momentum SGD step size: {best_lr} (test accuracy {best_acc})")



def run_momentum_sgd_hyperparameter_grid(train_dataset, test_dataset):
	loss_fn = torch.nn.CrossEntropyLoss()
	hidden_dims = [256, 512, 1024]
	num_hidden_layers_options = [1, 2, 3]
	momentum_options = [0.7, 0.8, 0.9, 0.99]
	results = []
	best_acc = float("-inf")
	best_config = None

	for hidden_dim in hidden_dims:
		for num_hidden_layers in num_hidden_layers_options:
			for momentum in momentum_options:
				print(f"Config: width={hidden_dim}, layers={num_hidden_layers}, momentum={momentum}")
				model = make_fully_connected_model_custom_part2_2(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers)
				optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=momentum)
				train_dataloader, test_dataloader = construct_dataloaders(train_dataset, test_dataset, 100)
				start_time = _sync_and_retrieve_time()
				stats = train(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs=10,eval_test_stats=False,eval_train_stats=False)
				wall_time = _sync_and_retrieve_time() - start_time
				test_loss,test_acc = evaluate_model(test_dataloader,model,loss_fn)
				config_result = {
					"hidden_dim": hidden_dim,
					"num_hidden_layers": num_hidden_layers,
					"momentum": momentum,
					"test_loss": test_loss,
					"test_acc": test_acc,
					"wall_time": wall_time,
				}
				results.append(config_result)
				filename = f"stats_grid_momentum_sgd_width_{hidden_dim}_layers_{num_hidden_layers}_momentum_{momentum}.pkl"
				with open(filename, "wb") as f:
					pickle.dump(stats, f)
				print(f"  test_acc={test_acc}, test_loss={test_loss}, wall_time={wall_time}s")
				if test_acc is not None and test_acc > best_acc:
					best_acc = test_acc
					best_config = config_result
	print(best_config)
	return results, best_config


def run_random_momentum_sgd_hyperparameter(train_dataset, test_dataset):
	loss_fn = torch.nn.CrossEntropyLoss()
	hidden_dims = (1024 - 256) * torch.rand(10) + 256
	num_hidden_layers_options = (3 - 1) * torch.rand(10) + 1
	momentum_options = (0.99 - 0.7) * torch.rand(10) + 0.7
	results = []
	best_acc = float("-inf")
	best_config = None


	for i in range(10):
		hidden_dim = int(hidden_dims[i])
		num_hidden_layers = int(num_hidden_layers_options[i])
		momentum = momentum_options[i]			
		print(f"Config: width={hidden_dim}, layers={num_hidden_layers}, momentum={momentum}")
		model = make_fully_connected_model_custom_part2_2(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers)
		optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=momentum)
		train_dataloader, test_dataloader = construct_dataloaders(train_dataset, test_dataset, 100)
		start_time = _sync_and_retrieve_time()
		stats = train(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs=10,eval_test_stats=False,eval_train_stats=False)
		wall_time = _sync_and_retrieve_time() - start_time
		test_loss,test_acc = evaluate_model(test_dataloader,model,loss_fn)
		config_result = {
			"hidden_dim": hidden_dim,
			"num_hidden_layers": num_hidden_layers,
			"momentum": momentum,
			"test_loss": test_loss,
			"test_acc": test_acc,
			"wall_time": wall_time,
		}
		results.append(config_result)
		filename = f"stats_random_momentum_sgd_width_{hidden_dim}_layers_{num_hidden_layers}_momentum_{momentum}.pkl"
		with open(filename, "wb") as f:
			pickle.dump(stats, f)
		print(f"  test_acc={test_acc}, test_loss={test_loss}, wall_time={wall_time}s")
		if test_acc is not None and test_acc > best_acc:
			best_acc = test_acc
			best_config = config_result
	print(best_config)
	return results, best_config

def run_3_1(train_dataset,test_dataset):
	loss_fn = torch.nn.CrossEntropyLoss()
	model = make_cnn_model_part3_1()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999))
	train_dataloader,test_dataloader = construct_dataloaders(train_dataset,test_dataset,100)
	start_time = _sync_and_retrieve_time()
	stats = train(train_dataloader,test_dataloader,model,loss_fn,optimizer,epochs=10)
	wall_time = _sync_and_retrieve_time() - start_time
	with open("stats_3_1.pkl","wb") as f:
		pickle.dump(stats,f)
	print("test accuracy: " + str(stats[3][-1]))
	print(f"training wall time: {wall_time}s")

def plot_graphs():
	model_runs = {
		"1.1": "stats_1.1.pkl",
		"1.2": "stats_1.2.pkl",
		"1.3": "stats_1.3.pkl",
		"1.4": "stats_1.4.pkl",
	}

	os.makedirs("figures", exist_ok=True)

	for name, filename in model_runs.items():
		with open(filename, "rb") as f:
			stats = pickle.load(f)
		assert len(stats) == 6
		train_loss, train_acc, test_loss, test_acc, approx_tr_loss, approx_tr_acc = stats
		epochs_range = list(range(1, len(approx_tr_loss) + 1))

		fig_loss = pyplot.figure()
		pyplot.plot(epochs_range, approx_tr_loss, label="Approx train loss")
		pyplot.plot(epochs_range, train_loss, label="Train loss")
		pyplot.plot(epochs_range, test_loss, label="Test loss")
		pyplot.xlabel("Epoch")
		pyplot.ylabel("Loss")
		pyplot.title(f"Loss Curves Run {name}")
		pyplot.legend()
		pyplot.savefig(f"figures/loss_curves_{name}.png")
		pyplot.close(fig_loss)

		fig_acc = pyplot.figure()
		pyplot.plot(epochs_range, approx_tr_acc, label="Approx train acc")
		pyplot.plot(epochs_range, train_acc, label="Train acc")
		pyplot.plot(epochs_range, test_acc, label="Test acc")
		pyplot.xlabel("Epoch")
		pyplot.ylabel("Accuracy")
		pyplot.title(f"Accuracy Curves Run {name}")
		pyplot.legend()
		pyplot.savefig(f"figures/accuracy_curves_{name}.png")
		pyplot.close(fig_acc)

def plot_run_3_1_figures():
	with open("stats_3_1.pkl", "rb") as f:
		stats = pickle.load(f)
	if len(stats) < 6:
		raise ValueError("Expected at least six elements in stats for run 3.1.")
	train_loss, train_acc, test_loss, test_acc, approx_tr_loss, approx_tr_acc = stats[:6]
	epochs_train = list(range(1, len(approx_tr_loss) + 1))
	epochs_test = list(range(1, len(test_loss) + 1)) if test_loss else []
	epochs_test_acc = list(range(1, len(test_acc) + 1)) if test_acc else []

	os.makedirs("figures", exist_ok=True)

	fig_loss = pyplot.figure()
	pyplot.plot(epochs_train, approx_tr_loss, label="Train loss (minibatch-derived)")
	if test_loss:
		pyplot.plot(epochs_test, test_loss, label="Test loss")
	pyplot.xlabel("Epoch")
	pyplot.ylabel("Loss")
	pyplot.title("Run 3.1 Loss vs Epoch")
	pyplot.legend()
	pyplot.savefig("figures/run_3_1_loss.png")
	pyplot.close(fig_loss)

	fig_acc = pyplot.figure()
	pyplot.plot(epochs_train, approx_tr_acc, label="Train accuracy (minibatch-derived)")
	if test_acc:
		pyplot.plot(epochs_test_acc, test_acc, label="Test accuracy")
	pyplot.xlabel("Epoch")
	pyplot.ylabel("Accuracy")
	pyplot.title("Run 3.1 Accuracy vs Epoch")
	pyplot.legend()
	pyplot.savefig("figures/run_3_1_accuracy.png")
	pyplot.close(fig_acc)

if __name__ == "__main__":
	(train_dataset, test_dataset) = load_MNIST_dataset()
	#run_1_1(train_dataset,test_dataset)
	#run_1_2(train_dataset,test_dataset)
	#run_1_3(train_dataset,test_dataset)
	#run_1_4(train_dataset,test_dataset)

	run_momentum_sgd_grid_search(train_dataset,test_dataset)
	run_momentum_sgd_hyperparameter_grid(train_dataset,test_dataset)
	run_random_momentum_sgd_hyperparameter(train_dataset,test_dataset)
	#plot_graphs()
