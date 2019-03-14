import time 
import model
import data_loader
from optimizer import optimizer_
import torch
from torch.backends import cudnn
from tqdm import tqdm


def train_(model_ = model.Net()):


	train_loader,valid_loader= data_loader.get_train_valid_loader(data_dir='./data',batch_size=16,augment=False,random_seed=1230,show_sample=False,num_workers=6,pin_memory=False)
	
	
	cudnn.benchmark = True
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('Using Device: %s' %device)
	net = model_
	net = net.cuda()
	optimizer, criterion = optimizer_(net)
	start_time = time.time()





	total_epoch = 3
	for epoch in tqdm(range(total_epoch)):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(train_loader, 0):
			# get the inputs
			inputs, labels = data
			inputs, labels = inputs.cuda(), labels.cuda()

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.cuda()
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % 1000 == 999:    # print every 1000 mini-batches
				correct = 0
				total = 0
				with torch.no_grad():
					valid_running_loss = 0 
					for data in valid_loader:
						images, labels = data
						images, labels = images.to(device), labels.to(device)
						outputs = net(images)

						valid_loss = criterion(outputs, labels)
						valid_loss.cuda()
						valid_running_loss += valid_loss.item()

						_, predicted = torch.max(outputs.data, 1)
						total += labels.size(0)
						correct += (predicted == labels).sum().item()

				print('[%d, %5d] loss: %.3f ' %
					  (epoch + 1, i + 1, running_loss / 1000))
				print('        Validation Set, loss = %.3f, Acc.: %.2f' %(valid_running_loss/len(valid_loader), correct/total))
				running_loss = 0.0
				valid_running_loss = 0
	end_time = time.time()
	print('Finished Training')
	print("Duration : {:d}".format(end_time-start_time))
	return net
	
	
	
from eval import eval_	
	
if __name__ == '__main__':
	net = train_()
	eval_(net)
	