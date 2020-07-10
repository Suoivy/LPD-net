import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import random

#####For training and test data split#####
x_width=150
y_width=150

#For Oxford
p1=[5735712.768124,620084.402381]
p2=[5735611.299219,620540.270327]
p3=[5735237.358209,620543.094379]
p4=[5734749.303802,619932.693364]   

#For University Sector
p5=[363621.292362,142864.19756]
p6=[364788.795462,143125.746609]
p7=[363597.507711,144011.414174]

#For Residential Area
p8=[360895.486453,144999.915143]
p9=[362357.024536,144894.825301]
p10=[361368.907155,145209.663042]

p_dict={"oxford":[p1,p2,p3,p4], "university":[p5,p6,p7], "residential": [p8,p9,p10], "business":[]}

def check_in_test_set(northing, easting, points, x_width, y_width):
	in_test_set=False
	for point in points:
		if(point[0]-x_width<northing and northing< point[0]+x_width and point[1]-y_width<easting and easting<point[1]+y_width):
			in_test_set=True
			break
	return in_test_set
##########################################

def output_to_file(output, filename):
	with open(filename, 'wb') as handle:
	    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("Done ", filename)


def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, p, output_name):

	database_sets=[]
	for folder in folders:
		database={}
		df_locations= pd.read_csv(os.path.join(base_path,runs_folder,folder,filename),sep=',')
		df_locations['timestamp']=runs_folder+folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
		df_locations=df_locations.rename(columns={'timestamp':'file'})
		for index,row in df_locations.iterrows():
			database[len(database.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
		database_sets.append(database)

	output_to_file(database_sets, output_name+'_inference_database.pickle')


###Building database and query files for evaluation
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path= "../../benchmark_datasets/"

folders=[]
index_list=[0]    # Need to be modified to find the folder you want (folders for name and index_list for index)
runs_folder = "oxford/"
featurecloud_fols = "/featurecloud_20m_10overlap/"
all_folders=sorted(os.listdir(os.path.join(BASE_DIR,base_path,runs_folder)))

for index in index_list:
	folders.append(all_folders[index])

print(folders)
construct_query_and_database_sets(base_path, runs_folder, folders, featurecloud_fols, "pointcloud_locations_20m_10overlap.csv", p_dict["oxford"], "oxford")

'''
#For Oxford
folders=[]
runs_folder = "oxford/"
featurecloud_fols = "/featurecloud_20m/"
all_folders=sorted(os.listdir(os.path.join(BASE_DIR,base_path,runs_folder)))
index_list=[5,6,7,9,10,11,12,13,14,15,16,17,18,19,22,24,31,32,33,38,39,43,44]
print(len(index_list))
for index in index_list:
	folders.append(all_folders[index])

print(folders)
construct_query_and_database_sets(base_path, runs_folder, folders, featurecloud_fols, "pointcloud_locations_20m.csv", p_dict["kitti"], "oxford")

#For University Sector
folders=[]
runs_folder = "inhouse_datasets/"
all_folders=sorted(os.listdir(os.path.join(BASE_DIR,base_path,runs_folder)))
uni_index=range(10,15)
for index in uni_index:
	folders.append(all_folders[index])

print(folders)
construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/", "pointcloud_centroids_25.csv", p_dict["university"], "university")

#For Residential Area
folders=[]
runs_folder = "inhouse_datasets/"
all_folders=sorted(os.listdir(os.path.join(BASE_DIR,base_path,runs_folder)))
res_index=range(5,10)
for index in res_index:
	folders.append(all_folders[index])

print(folders)
construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/", "pointcloud_centroids_25.csv", p_dict["residential"], "residential")

#For Business District
folders=[]
runs_folder = "inhouse_datasets/"
all_folders=sorted(os.listdir(os.path.join(BASE_DIR,base_path,runs_folder)))
bus_index=range(5)
for index in bus_index:
	folders.append(all_folders[index])

print(folders)
construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/", "pointcloud_centroids_25.csv", p_dict["business"], "business")
'''
