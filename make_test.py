import os
import utils
import random

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

valid_size = 1414

size = len(trainingList)
valid_list = []
while True:
    r = random.randint(0, size-1)
    pth = trainingList.pop(r)
    valid_list.append(pth)
    size -= 1
    if len(trainingList) == 12731:
        break

utils.list_to_file([trainingList], 'data/rob_ben_train.txt')
utils.list_to_file([valid_list], 'data/rob_ben_valid.txt')

print(len(trainingList), len(valid_list))
