import os
import csv
import json
import numpy as np


with open('./datasets/places205/train_places205.csv') as f:
	train_list = [[] for _ in range(205)]
	for line in csv.reader(f):
		file_name, class_id = line[0].split()
		train_list[int(class_id)].append(file_name)


with open('./datasets/places205/val_places205.csv') as f:
	val_list = [[] for _ in range(205)]
	for line in csv.reader(f):
		file_name, class_id = line[0].split()
		val_list[int(class_id)].append(file_name)



class_names = ['_'.join(val_list[i][0].split('/')[1:-1]) for i in range(205)]

base_dir = './datasets/places205/data/vision/torralba/deeplearning/images256'
target_dir = './datasets/imagenet-100-places-205'


np.random.seed(42)
for class_id, class_name in enumerate(class_names):
	target_folder = os.path.join(target_dir, 'val', 'place_%s'%class_name)
	os.makedirs(target_folder, exist_ok=True)
	selected_file_list = np.array(val_list[class_id])[np.random.permutation(np.arange(100))[:50]]
	for file_name in selected_file_list:
		os.system('ln -s %s %s'%(os.path.join(base_dir, file_name), os.path.join(target_folder, file_name.split('/')[-1])))


np.random.seed(42)
for class_id, class_name in enumerate(class_names):
	target_folder = os.path.join(target_dir, 'train', 'place_%s'%class_name)
	os.makedirs(target_folder, exist_ok=True)
	n_sample = len(train_list[class_id])
	selected_file_list = np.array(train_list[class_id])[np.random.permutation(np.arange(n_sample))[:1300]]
	for file_name in selected_file_list:
		os.system('ln -s %s %s'%(os.path.join(base_dir, file_name), os.path.join(target_folder, file_name.split('/')[-1])))

