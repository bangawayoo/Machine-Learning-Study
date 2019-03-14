import data_loader
import model
from optimizer import optimizer_
import torch

def eval_(model_):
	test_loader = data_loader.get_test_loader(data_dir='./data',batch_size=16,num_workers=4,pin_memory=False)
	net = model_
	net = net.cuda()
	optimizer, criterion = optimizer_(net)

	correct = 0
	total = 0
	with torch.no_grad():
		for data in test_loader:
			images, labels = data
			images, labels = images.cuda(), labels.cuda()
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (
		100 * correct / total))