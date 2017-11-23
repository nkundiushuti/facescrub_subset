import os,glob
import scipy
from scipy.misc import imread
from scipy.misc import imresize
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

#download Facescrub cropped from the official Megaface page: http://megaface.cs.washington.edu/participate/challenge.html and unzip in into path_to_facescrub

maxPeople=20 #take a subset of the people with the most photos
path_to_facescrub = '/Volumes/HDD 1/Downloads/facescrub_aligned/'

people = sorted([name for name in os.listdir(path_to_facescrub) if os.path.isdir(os.path.join(path_to_facescrub, name))])

fileCounter = []
for actid,act in enumerate(people):
    fileCounter.append(len(glob.glob1(os.path.join(path_to_facescrub,act),"*.png")))

fileCounter=np.array(fileCounter)
#selected = np.argsort(fileCounter)[::-1][:maxPeople]
selected = sorted(range(len(fileCounter)), key=lambda i: fileCounter[i], reverse=True)[:maxPeople]
countall = np.sum(fileCounter[selected])

images=np.zeros((countall,3,32,32),dtype=np.int8)
labels=np.zeros(countall,dtype=np.int8)
k = 0
i = 0
for actid,act in enumerate(people):
    if actid in selected:
        for img in glob.glob1(os.path.join(path_to_facescrub,act),"*.png"):
            rawimg = imread(os.path.join(path_to_facescrub,act,img),mode='RGB')
            images[k] = imresize(rawimg,(32,32)).reshape((3,32,32))
            labels[k] = i
            k = k + 1
        i = i + 1

images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.1, random_state=42)

train={}
train['features'] = images_train
train['labels'] = labels_train
test={}
test['features'] = images_test
test['labels'] = labels_test

with open(os.path.join(path_to_facescrub,'facescrub_train_'+str(maxPeople)+'.pkl'), 'wb') as output:
    pickle.dump(train, output, pickle.HIGHEST_PROTOCOL)

with open(os.path.join(path_to_facescrub,'facescrub_test_'+str(maxPeople)+'.pkl'), 'wb') as output:
    pickle.dump(test, output, pickle.HIGHEST_PROTOCOL)
