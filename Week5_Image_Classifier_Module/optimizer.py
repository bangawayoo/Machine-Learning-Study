import torch.optim as optim
import torch.nn as nn
def optimizer_(net,lr=0.001,momentum=0.9):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	return optimizer, criterion