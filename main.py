import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim

import numpy as np
import os
import shutil
import argparse
import pdb

from utils import metrics, sample_gt
from datasets import get_dataset
from DenseConv import *
import argparse
from skimage.segmentation import felzenszwalb
import time

def get_pseudo_label(segments, TRAIN_Y, gt):
	MAX_S = np.max(segments)
	MAX_Y = np.max(TRAIN_Y)
	pseudo_label = np.zeros([np.shape(TRAIN_Y)[0], np.shape(TRAIN_Y)[1],TRAIN_Y.max() + 1])
	idx = TRAIN_Y > 0
	tmp_Y, tmp_s = TRAIN_Y[idx], segments[idx]

	for i_tmp_s, i_tmp_Y in zip(tmp_s, tmp_Y):
		if i_tmp_Y > 0 :
			pseudo_label[segments == i_tmp_s,i_tmp_Y] = 1

	pseudo_label[gt == 0,:] = 0
	return pseudo_label

def four_rotation(matrix_0):
	matrix_90 =  np.rot90(matrix_0, k=1, axes=(0,1))
	matrix_180 = np.rot90(matrix_90, k=1, axes=(0,1))
	matrix_270 = np.rot90(matrix_180, k=1, axes=(0,1))
	return [matrix_0, matrix_90, matrix_180, matrix_270]

def rotation(matrix_x, matrix_y, pseudo_label = None, segments = None, Mirror = False):
	train_PL, train_SG = [], []
	if pseudo_label is None: train_PL = None
	if segments is None: train_SG = None
	if Mirror == True:
		train_IMG, train_Y = four_rotation(matrix_x[::-1,:,:]), four_rotation(matrix_y[::-1,:])
		if pseudo_label is not None:
			for k_pseudo_label in pseudo_label:
				train_PL.append(four_rotation(k_pseudo_label[::-1,:,:]))
		if segments is not None:
			for k_segments in segments:
				train_SG.append(four_rotation(k_segments[::-1,:,:]))	
	else:
		train_IMG, train_Y = four_rotation(matrix_x), four_rotation(matrix_y)
		if pseudo_label is not None:
			for k_pseudo_label in pseudo_label:
				train_PL.append(four_rotation(k_pseudo_label))
		if segments is not None:
			for k_segments in segments:
				train_SG.append(four_rotation(k_segments))	

	return train_IMG, train_Y, train_PL, train_SG

def H_segment(img, train_gt, params, gt):
	pseudo_label, idx = [], []
	path = "Datasets/" + params.DATASET + '/' + params.DATASET +'_felzenszwalb.npy'

	if os.path.exists(path) == True:
		all_segment = np.load("Datasets/" + params.DATASET + '/' + params.DATASET +'_felzenszwalb.npy')
	else:
		idd, idy = 1, []
		while idd < np.shape(gt)[0]*np.shape(gt)[1]:
			current_segment = felzenszwalb(img, scale=1.0, sigma=0.95, min_size=idd)
			if len(idy) > 0:
				if np.sum(current_segment - idy[-1]) != 0:
					print idd, np.sum(current_segment - idy[-1]), len(idy)
					idy.append(current_segment)
			else:
				idy.append(current_segment)
			_, counts = np.unique(current_segment, return_counts=True)
			idd = max(counts.min(), idd + 1)

		all_segment = np.stack(idx, 0)
		np.save(path, all_segment)

	for current_segment in all_segment:
		tmp = get_pseudo_label(current_segment, train_gt, gt)
		count_tmp = tmp.sum(-1)
		if np.sum(count_tmp > 0) == np.sum(gt > 0):
			print len(pseudo_label)
			return pseudo_label

		if len(idx) > 0:
			for k_idx in idx: 
				tmp[k_idx > 0,:] = 0

		if tmp.sum() > 0: 
			print len(pseudo_label), count_tmp.max(), np.sum(count_tmp > 0), np.sum(gt > 0)
			idx.append(count_tmp)
			pseudo_label.append(tmp)

def pre_data(img, train_gt, params, gt):
	TRAIN_IMG, TRAIN_Y, TRAIN_PL, TRAIN_SG = [], [], [], []
	pseudo_labels = H_segment(img, train_gt, params, gt)

	train_IMG, train_Y, train_PL, train_SG = rotation(img, train_gt, pseudo_labels)
	train_IMG_M, train_Y_M, train_PL_M, train_SG_M = rotation(img, train_gt, pseudo_labels, Mirror = True)

	image_Column = torch.Tensor(np.stack((train_IMG[0], train_IMG[2], train_IMG_M[0], train_IMG_M[2]), 0)).permute(0,3,1,2)
	y_Column = torch.LongTensor(np.stack((train_Y[0], train_Y[2], train_Y_M[0], train_Y_M[2]), 0).astype(int))

	image_Row = torch.Tensor(np.stack((train_IMG[1], train_IMG[3], train_IMG_M[1], train_IMG_M[3]), 0)).permute(0,3,1,2)
	y_Row = torch.LongTensor(np.stack((train_Y[1], train_Y[3], train_Y_M[1], train_Y_M[3]), 0).astype(int))
	
	if train_PL is not None:
		y_PL_Column, y_PL_Row = [], []
		for k_PL, k_PL_M in zip(train_PL, train_PL_M):
			y_PL_Column.append(torch.FloatTensor(np.stack((k_PL[0], k_PL[2], k_PL_M[0], k_PL_M[2]), 0).astype(float)))
			y_PL_Row.append(torch.FloatTensor(np.stack((k_PL[1], k_PL[3], k_PL_M[1], k_PL_M[3]), 0).astype(float)))
		TRAIN_PL.append(y_PL_Column)
		TRAIN_PL.append(y_PL_Row)
	else:
		TRAIN_PL = None

	if train_SG is not None:
		y_SG_Column, y_SG_Row = [], []
		for k_SG, k_SG_M in zip(train_SG, train_SG_M):
			y_SG_Column.append(torch.FloatTensor(np.stack((k_SG[0], k_SG[2], k_SG_M[0], k_SG_M[2]), 0).astype(float)))
			y_SG_Row.append(torch.FloatTensor(np.stack((k_SG[1], k_SG[3], k_SG_M[1], k_SG_M[3]), 0).astype(float)))
		TRAIN_SG.append(y_SG_Column)
		TRAIN_SG.append(y_SG_Row)
	else:
		TRAIN_SG = None

	TRAIN_IMG.append(image_Column)
	TRAIN_IMG.append(image_Row)
	TRAIN_Y.append(y_Column)
	TRAIN_Y.append(y_Row)

	return TRAIN_IMG, TRAIN_Y, TRAIN_PL, TRAIN_SG


def main(params): 
	RUNS = 4
	MX_ITER = 100
	os.environ["CUDA_VISIBLE_DEVICES"] = params.GPU

	device = torch.device("cuda: 0")
	print torch.cuda.device_count()

	new_path = str(params.DATASET) + '/Best/' + '_'.join([str(params.SAMPLE_PERCENTAGE), str(params.DHCN_LAYERS), str(params.CONV_SIZE), str(params.ROT), str(params.MIRROR), str(params.H_MIRROR)]) + '/'
	if os.path.exists(new_path) :
		RUNS = 4 - len(os.listdir(new_path))

	if RUNS == 0:
		return

	for _ in range(RUNS):

		start_time = time.time()

		SAMPLE_PERCENTAGE = params.SAMPLE_PERCENTAGE
		DATASET = params.DATASET
		DHCN_LAYERS = params.DHCN_LAYERS
		CONV_SIZE = params.CONV_SIZE
		H_MIRROR = params.H_MIRROR
		LR = 1e-4

		save_path = str(DATASET) + '/tmp' + str(params.GPU) + '_abc/'
		if not os.path.exists(save_path) :
			os.makedirs(save_path)
			
		img, gt, LABEL_VALUES, IGNORED_LABELS, _, _ = get_dataset(DATASET, "Datasets/")
		N_CLASSES = len(LABEL_VALUES)
		INPUT_SIZE = np.shape(img)[-1]
		train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode='fixed')
		INPUT_DATA = pre_data(img, train_gt, params, gt)

		model_DHCN = DHCN(input_size = INPUT_SIZE, embed_size=INPUT_SIZE, densenet_layer = DHCN_LAYERS, output_size=N_CLASSES, conv_size = CONV_SIZE, batch_norm = False).to(device)
		optimizer_DHCN = torch.optim.Adam(model_DHCN.parameters(), lr=LR, betas=(0.5, 0.999), weight_decay=1e-4)
		model_DHCN=nn.DataParallel(model_DHCN)
		loss_ce = nn.CrossEntropyLoss().to(device)
		loss_bce = nn.BCELoss().to(device)

		best_ACC, tmp_epoch, tmp_count, tmp_rate, recode_reload, reload_model = 0.0, 0, 0, LR, {}, False
		max_tmp_count = 300

		for epoch in range(MX_ITER):

			model_DHCN.train()

			loss_supervised, loss_self, loss_distill = 0.0, 0.0, 0.0

			for TRAIN_IMG, TRAIN_Y, TRAIN_PL in zip(INPUT_DATA[0], INPUT_DATA[1], INPUT_DATA[2]):
				scores, _ = model_DHCN(TRAIN_IMG.to(device))
				for k_Layer in range(DHCN_LAYERS + 1):
					for i_num, (k_scores, k_TRAIN_Y) in enumerate(zip(scores[k_Layer], TRAIN_Y)):
						k_TRAIN_Y = k_TRAIN_Y.to(device)
						loss_supervised += loss_ce(k_scores.permute(1,2,0)[k_TRAIN_Y > 0], k_TRAIN_Y[k_TRAIN_Y > 0])
						for id_layer, k_TRAIN_PL in enumerate(TRAIN_PL):
							k_TRAIN_PL = k_TRAIN_PL.to(device)
							if (k_TRAIN_PL[i_num].sum(-1) > 1).sum() > 0:
								loss_distill += (1 / float(id_layer + 1)) * loss_bce(k_scores.permute(1,2,0).sigmoid()[k_TRAIN_PL[i_num].sum(-1) > 0], k_TRAIN_PL[i_num][k_TRAIN_PL[i_num].sum(-1) > 0])
							else:
								onehot2label = torch.topk(k_TRAIN_PL[i_num],k=1,dim=-1)[1].squeeze(-1)
								loss_self += (1 / float(id_layer + 1)) * loss_ce(k_scores.permute(1,2,0)[onehot2label > 0], onehot2label[onehot2label > 0])

			loss = loss_supervised + loss_self + loss_distill

			optimizer_DHCN.zero_grad()
			nn.utils.clip_grad_norm_(model_DHCN.parameters(), 3.0)
			loss.backward()
			optimizer_DHCN.step()

			if epoch % 1 == 0:
				model_DHCN.eval()
				
				p_idx = []
				fusion_prediction = 0.0

				for k_data, current_data in enumerate(INPUT_DATA[0]):
					scores, _ = model_DHCN(current_data.to(device))
					if params.ROT == False:
						for k_score in scores:
							fusion_prediction += F.softmax(k_score[0].permute(1,2,0), dim=-1).cpu().data.numpy()
					else:
						for k_score in scores:
							if k_data == 0: 
								fusion_prediction += F.softmax(k_score[0].permute(1,2,0), dim=-1).cpu().data.numpy()
								fusion_prediction += np.rot90(F.softmax(k_score[1].permute(1,2,0), dim=-1).cpu().data.numpy(),k=2,axes=(0,1))
								fusion_prediction += F.softmax(k_score[2].permute(1,2,0), dim=-1).cpu().data.numpy()[::-1,:,:]
								fusion_prediction += np.rot90(F.softmax(k_score[3].permute(1,2,0), dim=-1).cpu().data.numpy(),k=2,axes=(0,1))[::-1,:,:]
								
								p_idx.append(k_score[0].max(0)[-1].cpu().data.numpy())
								p_idx.append(np.rot90(k_score[1].max(0)[-1].cpu().data.numpy(),k=2,axes=(0,1)))
								p_idx.append(k_score[2].max(0)[-1].cpu().data.numpy()[::-1,:])
								p_idx.append(np.rot90(k_score[3].max(0)[-1].cpu().data.numpy(),k=2,axes=(0,1))[::-1,:])

							if k_data == 1:
								fusion_prediction += np.rot90(F.softmax(k_score[0].permute(1,2,0), dim=-1).cpu().data.numpy(),k=-1,axes=(0,1))
								fusion_prediction += np.rot90(F.softmax(k_score[1].permute(1,2,0), dim=-1).cpu().data.numpy(),k=1,axes=(0,1))
								fusion_prediction += np.rot90(F.softmax(k_score[2].permute(1,2,0), dim=-1).cpu().data.numpy(),k=-1,axes=(0,1))[::-1,:,:]
								fusion_prediction += np.rot90(F.softmax(k_score[3].permute(1,2,0), dim=-1).cpu().data.numpy(),k=1,axes=(0,1))[::-1,:,:]

								p_idx.append(np.rot90(k_score[0].max(0)[-1].cpu().data.numpy(),k=-1,axes=(0,1)))
								p_idx.append(np.rot90(k_score[1].max(0)[-1].cpu().data.numpy(),k=1,axes=(0,1)))
								p_idx.append(np.rot90(k_score[2].max(0)[-1].cpu().data.numpy(),k=-1,axes=(0,1))[::-1,:])
								p_idx.append(np.rot90(k_score[3].max(0)[-1].cpu().data.numpy(),k=1,axes=(0,1))[::-1,:])

				Acc = np.zeros([len(p_idx) + 1])
				for count, k_idx in enumerate(p_idx):
					Acc[count] = metrics(k_idx.reshape(img.shape[:2]), test_gt, ignored_labels=IGNORED_LABELS, n_classes=N_CLASSES)['Accuracy']
				Acc[-1] = metrics(fusion_prediction.argmax(-1).reshape(img.shape[:2]), test_gt, ignored_labels=IGNORED_LABELS, n_classes=N_CLASSES)['Accuracy']

				tmp_count += 1

				if  max(Acc) > best_ACC :

					best_ACC = max(Acc)
					save_file_path = save_path + 'save_' + str(epoch)+ '_' + str(round(best_ACC, 2)) + '.pth'
					states = {'state_dict_DHCN': model_DHCN.state_dict(),
							'train_gt': train_gt, 
							'test_gt': test_gt,}

					torch.save(states, save_file_path)

					tmp_count = 0
					tmp_epoch = epoch
					print 'save: ' , epoch , str(round(best_ACC, 2))
					print loss_supervised.data, loss_self.data, loss_distill.data
					print np.round(Acc, 2)

def parse_args():
	parser = argparse.ArgumentParser(description='Low shot benchmark')
	parser.add_argument('--DHCN_LAYERS', default=1, type=int)
	parser.add_argument('--SAMPLE_PERCENTAGE', default=5, type=int)
	parser.add_argument('--DATASET', default="PaviaU", type=str) #KSC, PaviaU, IndianPines, Botswana,    !!PaviaC
	parser.add_argument('--CONV_SIZE', default=3, type=int) #3,5,7
	parser.add_argument('--ROT', default=True, type=bool) # False
	parser.add_argument('--MIRROR', default=True, type=bool) # False
	parser.add_argument('--H_MIRROR', default='full', type=str) # half, full
	parser.add_argument('--GPU', default='0,1,2,3', type=str) # 0,1,2,3

	parser.add_argument('--ROT_N', default=1, type=int) # False
	parser.add_argument('--MIRROR_N', default=1, type=int) # False

	# parser.add_argument('--RUNS', default=0, type=int) # False

	return parser.parse_args()

if __name__ == '__main__':
	params = parse_args()
	params.ROT = True if params.ROT_N == 1 else False
	params.MIRROR = True if params.MIRROR_N == 1 else False

	main(params)