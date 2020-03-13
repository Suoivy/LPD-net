'''
	Pre-processing: prepare_data in LPD-Net
	generate KNN neighborhoods and calculate feature as the feature matrix of point
	Reference: LPD-Net: 3D Point Cloud Learning for Large-Scale Place Recognition and Environment Analysis, ICCV 2019

	author: Chuanzhe Suo(suo_ivy@foxmail.com)
	created: 10/26/18
'''

import os
import sys
import multiprocessing as multiproc
from copy import deepcopy
import glog as logger
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KDTree
import math
import errno

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = "../benchmark_datasets/"

runs_folder= "oxford/"
filename = "pointcloud_locations_20m_10overlap.csv"
pointcloud_fols="pointcloud_20m_10overlap/"
featurecloud_fols="featurecloud_20m_10overlap/"


def calculate_features_old(pointcloud, nbrs_index, eigens):
	### calculate handcraft feature with eigens and statistics data

	# features using eigens
	eig3d = eigens['eigens'][:3]
	eig2d = eigens['eigens'][3:5]
	vetors = eigens['vectors']
	# 3d
	C_ = eig3d[2]/(eig3d.sum())
	O_ = np.power((eig3d.prod()/np.power(eig3d.sum(),3)),1.0/3)
	L_ = (eig3d[0]-eig3d[1])/eig3d[0]
	E_ = -((eig3d/eig3d.sum())*np.log(eig3d/eig3d.sum())).sum()
	D_ = 3*nbrs_index.shape[0]/(4*math.pi*eig3d.prod())
	# 2d
	S_2 = eig2d.sum()
	L_2 = eig2d[1]/eig2d[0]
	# features using statistics data
	neighborhood = pointcloud[nbrs_index]
	nbr_dz = neighborhood[:,2]-neighborhood[:,2].min()
	dZ_ = nbr_dz.max()
	vZ_ = np.var(nbr_dz)
	V_ = vetors[2][2]

	features = np.asarray([C_,O_,L_,E_,D_,S_2,L_2,dZ_,vZ_,V_])
	return features

def calculate_features(pointcloud, nbrs_index, eigens_, vectors_):
    ### calculate handcraft feature with eigens and statistics data

    # features using eigens
    eig3d = eigens_[:3]
    eig2d = eigens_[3:5]

    # 3d
    C_ = eig3d[2] / (eig3d.sum())
    O_ = np.power((eig3d.prod() / np.power(eig3d.sum(), 3)), 1.0 / 3)
    L_ = (eig3d[0] - eig3d[1]) / eig3d[0]
    E_ = -((eig3d / eig3d.sum()) * np.log(eig3d / eig3d.sum())).sum()
    #P_ = (eig3d[1] - eig3d[2]) / eig3d[0]
    #S_ = eig3d[2] / eig3d[0]
    #A_ = (eig3d[0] - eig3d[2]) / eig3d[0]
    #X_ = eig3d.sum()
    D_ = 3 * nbrs_index.shape[0] / (4 * math.pi * eig3d.prod())
    # 2d
    S_2 = eig2d.sum()
    L_2 = eig2d[1] / eig2d[0]
    # features using statistics data
    neighborhood = pointcloud[nbrs_index]
    nbr_dz = neighborhood[:, 2] - neighborhood[:, 2].min()
    dZ_ = nbr_dz.max()
    vZ_ = np.var(nbr_dz)
    V_ = vectors_[2][2]

    features = np.asarray([C_, O_, L_, E_,  D_, S_2, L_2, dZ_, vZ_, V_])#([C_,O_,L_,E_,D_,S_2,L_2,dZ_,vZ_,V_])
    return features

def calculate_entropy(eigen):
	L_ = (eigen[0] - eigen[1]) / eigen[0]
	P_ = (eigen[1] - eigen[2]) / eigen[0]
	S_ = eigen[2] / eigen[0]
	Entropy = -L_*np.log(L_)-P_*np.log(P_)-S_*np.log(S_)
	return Entropy

def calculate_entropy_array(eigen):
    L_ = (eigen[:,0] - eigen[:,1]) / eigen[:,0]
    P_ = (eigen[:,1] - eigen[:,2]) / eigen[:,0]
    S_ = eigen[:,2] / eigen[:,0]
    Entropy = -L_*np.log(L_)-P_*np.log(P_)-S_*np.log(S_)
    return Entropy

def covariation_eigenvalue(neighborhood_index, args):
    ### calculate covariation and eigenvalue of 3D and 2D
    # prepare neighborhood
    neighborhoods = args.pointcloud[neighborhood_index]

    # 3D cov and eigen by matrix
    Ex = np.average(neighborhoods, axis=1)
    Ex = np.reshape(np.tile(Ex,[neighborhoods.shape[1]]), neighborhoods.shape)
    P = neighborhoods-Ex
    cov_ = np.matmul(P.transpose((0,2,1)),P)/(neighborhoods.shape[1]-1)
    eigen_, vec_ = np.linalg.eig(cov_)
    indices = np.argsort(eigen_)
    indices = indices[:,::-1]
    pcs_num_ = eigen_.shape[0]
    indx = np.reshape(np.arange(pcs_num_), [-1, 1])
    eig_ind = indices + indx*3
    vec_ind = np.reshape(eig_ind*3, [-1,1]) + np.full((pcs_num_*3,3), [0,1,2])
    vec_ind = np.reshape(vec_ind, [-1,3,3])
    eigen3d_ = np.take(eigen_, eig_ind)
    vectors_ = np.take(vec_, vec_ind)
    entropy_ = calculate_entropy_array(eigen3d_)

    # 2D cov and eigen
    cov2d_ = cov_[:,:2,:2]
    eigen2d, vec_2d = np.linalg.eig(cov2d_)
    indices = np.argsort(eigen2d)
    indices = indices[:, ::-1]
    pcs_num_ = eigen2d.shape[0]
    indx = np.reshape(np.arange(pcs_num_), [-1, 1])
    eig_ind = indices + indx * 2
    eigen2d_ = np.take(eigen2d, eig_ind)

    eigens_ = np.append(eigen3d_,eigen2d_,axis=1)

    return cov_, entropy_, eigens_, vectors_

def build_neighbors_NN(k, args):
	### using KNN NearestNeighbors cluster according k
	nbrs = NearestNeighbors(n_neighbors=k).fit(args.pointcloud)
	distances, indices = nbrs.kneighbors(args.pointcloud)
	covs, entropy, eigens_, vectors_ = covariation_eigenvalue(indices, args)
	neighbors = {}
	neighbors['k'] = k
	neighbors['indices'] = indices
	neighbors['covs'] = covs
	neighbors['entropy'] = entropy
	neighbors['eigens_'] = eigens_
	neighbors['vectors_'] = vectors_
	logger.info("KNN:{}".format(k))
	return neighbors

def build_neighbors_KDT(k, args):
	### using KNN KDTree cluster according k
	nbrs = KDTree(args.pointcloud)
	distances, indices = nbrs.query(args.pointcloud, k=k)
	covs, entropy, eigens_, vectors_ = covariation_eigenvalue(indices, args)
	neighbors = {}
	neighbors['k'] = k
	neighbors['indices'] = indices
	neighbors['covs'] = covs
	neighbors['entropy'] = entropy
	neighbors['eigens_'] = eigens_
	neighbors['vectors_'] = vectors_
	logger.info("KNN:{}".format(k))
	return neighbors


def prepare_file(pointcloud_file, args):
	### Parallel process pointcloud files
	# load pointcloud file
	pointcloud = np.fromfile(pointcloud_file, dtype=np.float64)
	pointcloud = np.reshape(pointcloud, (pointcloud.shape[0]//3, 3))
	args.pointcloud = pointcloud

	# prepare KNN cluster number k
	cluster_number = []
	for ind in range(((args.k_end-args.k_start)//args.k_step)+1):
		cluster_number.append(args.k_start + ind*args.k_step)

	k_nbrs = []
	for k in cluster_number:
		k_nbr = build_neighbors_NN(k, args)
		k_nbrs.append(k_nbr)

	logger.info("Processing pointcloud file:{}".format(pointcloud_file))
	# multiprocessing pool to parallel cluster pointcloud
	#pool = multiproc.Pool(len(cluster_number))
	#build_neighbors_func = partial(build_neighbors, args=deepcopy(args))
	#k_nbrs = pool.map(build_neighbors_func, cluster_number)
	#pool.close()
	#pool.join()

	# get argmin k according E, different points may have different k
	k_entropys = []
	for k_nbr in k_nbrs:
		k_entropys.append(k_nbr['entropy'])
	argmink_ind = np.argmin(np.asarray(k_entropys), axis=0)


	points_feature = []
	for index in range(pointcloud.shape[0]):
		### per point
		neighborhood = k_nbrs[argmink_ind[index]]['indices'][index]
		eigens_ = k_nbrs[argmink_ind[index]]['eigens_'][index]
		vectors_ = k_nbrs[argmink_ind[index]]['vectors_'][index]

		# calculate point feature
		feature = calculate_features(pointcloud, neighborhood, eigens_, vectors_)
		points_feature.append(feature)
	points_feature = np.asarray(points_feature)

	# save to point feature folders and bin files
	feature_cloud = np.append(pointcloud, points_feature, axis=1)
	pointfile_path, pointfile_name = os.path.split(pointcloud_file)
	filepath = os.path.join(os.path.split(pointfile_path)[0], args.featurecloud_fols, pointfile_name)
	feature_cloud.tofile(filepath)

	# build KDTree and store fot the knn query
	#kdt = KDTree(pointcloud, leaf_size=50)
	#treepath = os.path.splitext(filepath)[0] + '.pickle'
	#with open(treepath, 'wb') as handel:
	#	pickle.dump(kdt, handel)

	logger.info("Feature cloud file saved:{}".format(filepath))


def prepare_dataset(args):
	### Parallel process dataset folders
	# Initialize pandas DataFrame
	df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
	# load folder csv file
	df_locations = pd.read_csv(os.path.join(args.dataset_path, args.runs_folder, args.pointcloud_folder, args.filename), sep=',')
	df_locations['timestamp'] = args.base_path + args.runs_folder + args.pointcloud_folder +'/'+ args.pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin'
	df_locations = df_locations.rename(columns={'timestamp': 'file'})
	# creat feature_cloud folder
	featurecloud_path = os.path.join(args.dataset_path, args.runs_folder, args.pointcloud_folder, args.featurecloud_fols)
	if not os.path.exists(featurecloud_path):
		try:
			os.makedirs(featurecloud_path)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise
			pass

	pointcloud_files = df_locations['file'].tolist()
	#print(pointcloud_files)

	# multiprocessing pool to parallel process pointcloud_files
	pool = multiproc.Pool(args.bin_core_num)
	for file in pointcloud_files:
		file = os.path.join(args.BASE_DIR, file)
		pointfile_path, pointfile_name = os.path.split(file)
		filepath = os.path.join(os.path.split(pointfile_path)[0], args.featurecloud_fols, pointfile_name)
		if not os.path.exists(filepath):
			pool.apply_async(prepare_file,(file, deepcopy(args)))
		else:
			logger.info("{} exists, skipped".format(file))
	pool.close()
	logger.info("Cloud folder processing:{}".format(args.pointcloud_folder))
	pool.join()
	logger.info("end folder processing")

def run_all_processes(all_p):
	try:
		for p in all_p:
			p.start()
		for p in all_p:
			p.join()
	except KeyboardInterrupt:
		for p in all_p:
			if p.is_alive():
				p.terminate()
			p.join()
		exit(-1)


def main(args):
	# prepare dataset folders
	args.BASE_DIR = BASE_DIR
	args.base_path = base_path
	args.dataset_path = os.path.join(BASE_DIR, base_path)
	args.runs_folder = runs_folder
	args.pointcloud_fols = pointcloud_fols
	args.featurecloud_fols = featurecloud_fols
	args.filename = filename
	all_folders = sorted(os.listdir(os.path.join(BASE_DIR, base_path, runs_folder)))
	folders = []

	# All runs are used for training (both full and partial)
	index_list = range(len(all_folders)-1)
	print("Number of runs: " + str(len(index_list)))
	print(all_folders)
	for index in index_list:
		folders.append(all_folders[index])
	print(folders)

	# multiprocessing dataset folder
	all_p = []
	for folder in folders:
		args.pointcloud_folder = folder
		all_p.append(multiproc.Process(target=prepare_dataset, args=(deepcopy(args),)))
	run_all_processes(all_p)

	logger.info("Dataset preparation Finised")


if __name__ == '__main__':
	parse = argparse.ArgumentParser(sys.argv[0])

	parse.add_argument('--k_start', type=int, default=20,
	                   help="KNN cluster k range start point")
	parse.add_argument('--k_end', type=int, default=100,
	                   help="KNN cluster k range end point")
	parse.add_argument('--k_step', type=int, default=10,
	                   help="KNN cluster k range step")
	parse.add_argument('--bin_core_num', type=int, default=10, help="Parallel process file Pool core num")

	args = parse.parse_args(sys.argv[1:])
	main(args)
