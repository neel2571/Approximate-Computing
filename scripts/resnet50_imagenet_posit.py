import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from models_extension import alexnet,resnet18,densenet121,vgg16,vgg19,resnet50
#from models_extension_approx_first_layer import resnet18_layer1_approx,resnet18,LeNet5,LeNet5_layer1_approx
#from resnet18_model_parallel_pure_python import resnet18

best_prec1 = 0


def main():
	global  best_prec1

	# create model
	#model_urls['resnet18'] = model_urls['resnet18'].replace('https://', 'http://')
	#model = models.resnet18(pretrained=True)

	model = resnet50(pretrained=True)

	print(model.parameters())

 	#model.load_state_dict(torch.load('resnet18_actual.pth.tar'))
	#model.load_state_dict(torch.load('resnet18_8bit.pth.tar'))

	#model = model.cuda()

	# define loss function (criterion) and optimizer
	#criterion = nn.CrossEntropyLoss().cuda()
	criterion = nn.CrossEntropyLoss()

	# Data loading code
	valdir = os.path.join(os.getcwd(), '../val_5000/')

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

	val_loader = torch.utils.data.DataLoader(
		datasets.ImageFolder(valdir, transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=2, shuffle=False,
		num_workers=4, pin_memory=False)

	validate(val_loader, model, criterion)

def validate(val_loader, model, criterion):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(val_loader):
			#target = target.cuda(non_blocking=True)
			#input = input.cuda()
			# compute output
			input = Variable(input)
			output = model(input)
			loss = criterion(output, target)

			# measure accuracy and record loss
			prec1, prec5 = accuracy(output, target, topk=(1, 5))
			losses.update(loss.item(), input.size(0))
			top1.update(prec1[0], input.size(0))
			top5.update(prec5[0], input.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % 1 == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
					  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
					   i, len(val_loader), batch_time=batch_time, loss=losses,
					   top1=top1, top5=top5))

		print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
				  .format(top1=top1, top5=top5))
	#########################
	return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = 0.0001 * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.reshape(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


if __name__ == '__main__':
	main()
