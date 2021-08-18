import os
import utils

all_paths = 'data/robbins_benedix_paths.txt'
comb_paths = 'data/combined_test.txt'

with open(all_paths, 'r') as f:
    allList = f.readlines()

with open(comb_paths, 'r') as f:
    combList = f.readlines()

allList = [pth.rstrip(' \n') for pth in allList]
combList = [pth.rstrip(' \n') for pth in combList]
trainingList = [] 

for pth in allList:
    if pth not in combList:
        trainingList.append(pth)

print(len(trainingList))
print(trainingList[0:5])

utils.list_to_file([trainingList], 'data/rob_ben_train.txt')