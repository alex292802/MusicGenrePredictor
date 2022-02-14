#data preparation file for popularity prediction
import sys
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from random import randrange

np.set_printoptions(threshold=sys.maxsize)
sizeTest = 100
numOfIntervals = 10

#return the data and their labels in the form of a numpy array
def extractionDataset():
    data = pd.read_csv("./data/spotify_dataset_subset.csv")

    labels = data['popularity']
    #we drop populartity and ip
    #we also decide to drop artist_name and track_name
    data = data.drop(columns=['popularity','artist_name', 'track_name','id'])
    
    data = data.to_numpy()
    labels = labels.to_numpy()

    return data,labels

def computeIntervals(n=numOfIntervals):
    intervals = []
    j = 100/n
    i = j
    while i < 100:
        intervals.append(i)
        i += j
        
    intervals.append(100)

    return np.array(intervals)

def labelsIndexation(labels):
    intervals = computeIntervals()
    output = []
    for p in labels:
        counter = 0
        flag = False
        for i in intervals:
            if p <= i and not flag:   
                output.append(counter)
                flag = True
            counter+=1
            
    return np.array(output)

#args: data to be prepared
#return scalled dataset
def datasetPreparation(data):
    #features to be normalized : popularity, key, loundess, 
    #tempo, duration_ms, time_signature
    
    #convert all dates into years (and to int in the process)
    #change explicit values from boolean to int (0,1)
    #list all the genres
    genres = []

    for row in data:
        if len(row[0]) > 7:  
            row[0] = datetime.strptime(row[0], "%Y-%m-%d").year
        elif len(row[0]) > 4:
            row[0] = datetime.strptime(row[0], "%Y-%m").year
        else:
            row[0] = datetime.strptime(row[0], "%Y").year
        
        if row[2]:
            row[2] = 1
        else:
            row[2] = 0

        tmp = ast.literal_eval(row[1])
        for genre in tmp:
            genres.append(genre)

    genres = np.array(genres)
    unique, counts = np.unique(genres, return_counts=True)
    genresOccurences = dict(zip(unique, counts))

    #if there are multiple genres for one track we replace it by the one 
    #that occurs the most

    for row in data:
        tmp = ast.literal_eval(row[1])
        if len(tmp) > 1:
            maxOccurences = 0
            pick = ''
            for genre in tmp:
                occurences = genresOccurences[genre]
                if occurences > maxOccurences:
                    maxOccurences = occurences
                    pick = genre 

            row[1] = pick

        elif len(tmp) == 1:
            row[1] = tmp[0]

        else:
            row[1] = ''

    #we now generate an array containing all the genres once in order to index them
    genres = np.unique(data[:,1])

    #we know index all the genres
    #we have 713 different genres
    for row in data:
        row[1] = genres.tolist().index(row[1])

    #normalizing the values to (0,1)
    scalledData = np.empty(shape=(data.shape))
    for i in range(16):
        if i != 1:
            scaler = MinMaxScaler(feature_range=(0,1))
            column = data[:,i].reshape(-1,1)
            scaler.fit(column)
            scalledData[:,i] = scaler.transform(column)[:,0]
        else:
            scalledData[:,i] = data[:,i]

    return scalledData

#args:  scalled dataset 
#       array of labels indexes       
#       size of the testSet
#       mode : 0 - randomized testset picker
#              1 - pickes first n=size elements of data for testset
#return trainX data for training
#       trainY labels for training 
#       testX data for testing
#       testY labels for testing
def computeTestset(scalledData, labels, size=sizeTest, mode=0):
    testY = []
    testX = []
    
    trainX = []
    trainY = []

    if mode == 0: #Randomized test picker
        picked = []   
        for i in range(size):
            tmp = randrange(scalledData.shape[0])
            if tmp not in picked:
               picked.append(tmp)  
        
        c=0
        for i in scalledData:
            if c in picked:
                testX.append(i)
                testY.append(labels[c])
            else:
                trainX.append(i)
                trainY.append(labels[c])
            c+=1
        
    elif mode == 1: #pick first n=size elements
        c = 0
        for i in scalledData:
            if c > size:
                trainX.append(i)
                trainY.append(labels[c])
            else:
                testX.append(i)
                testY.append(labels[c])
            c+=1

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)

    return trainX, trainY, testX, testY

#args   size of the testSet
#       mode : 0 - randomized testset picker
#              1 - pickes first n=size elements of data for testset
#              
#return trainX data for training
#       trainY labels for training 
#       testX data for testing
#       testY labels for testing
def computeDatasets(sizeOfTestset = sizeTest, mode=1):
    data, labels = extractionDataset() 
    scalledData = datasetPreparation(data)
    labels = labelsIndexation(labels)
    return computeTestset(scalledData, labels, sizeOfTestset, mode)

def getOutputQty():
    return numOfIntervals

