import os
from os import listdir
from os.path import isfile, join

direc_path = "images/"
onlyfiles = [ f for f in listdir(direc_path) if isfile(join(direc_path,f)) ]

train_file = open("train_files.txt", "w")

for file in onlyfiles:
	name = file
	train_file.writelines("data/images/")
	train_file.writelines(name)
	train_file.writelines("\n")


train_file.close()