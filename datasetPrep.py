import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from random import randrange

sizeTest = 500

genres = ['r&b' ,'rap' ,'classical' ,'salsa' ,'edm' ,'hip hop', 'techno' 
,'jazz' ,'metal' ,'country' ,'rock' ,'reggae' , 'latin' ,'disco' ,'soul' ,'chanson' 
,'blues' ,'dance' ,'electro' ,'punk' ,'folk' ,'pop']


def getGenres():
    return genres

#return the data and their labels in the form of a numpy arrays
def extractionDataset():
    data = pd.read_csv("data/spotify_dataset_train.csv")

    labels = data['genre']
    data = data.drop(columns='genre')
    
    data = data.to_numpy()
    labels = labels.to_numpy()

    return data,labels

#return the challenge dataset in the form of a numpy array
def extractionChallenge():
    challenge = pd.read_csv("data/spotify_dataset_test.csv")
    challenge = challenge.to_numpy()

    return challenge

#args: data to be prepared
#return scalled dataset
def datasetPreparation(data):
    #features to be normalized : popularity, key, loundess, 
    #tempo, duration_ms, time_signature
    
    #convert all dates into years (and to int in the process)
    #change explicit values from boolean to int (0,1)
    for row in data:
        if len(row[0]) > 7:  
            row[0] = datetime.strptime(row[0], "%Y-%m-%d").year
        elif len(row[0]) > 4:
            row[0] = datetime.strptime(row[0], "%Y-%m").year
        else:
            row[0] = datetime.strptime(row[0], "%Y").year
        
        if row[1]:
            row[1] = 1
        else:
            row[1] = 0
    
    #normalizing the values to (0,1)
    scalledData = np.empty(shape=(data.shape))
    for i in range(16):
       scaler = MinMaxScaler(feature_range=(0,1))
       column = data[:,i].reshape(-1,1)
       scaler.fit(column)
       scalledData[:,i] = scaler.transform(column)[:,0]

    return scalledData

#args: array of labels in the form of strings
#return array of labels indexes
def labelsToIndex(labels):
    #replacing genres name by there respective indexes 
    labelsInt = []
    for row in labels:
        labelsInt.append(genres.index(row))
    
    return np.array(labelsInt)
    
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
def computeDatasets(sizeOfTestset = sizeTest, mode=0):
    data, labels = extractionDataset() 
    scalledData = datasetPreparation(data)
    labels = labelsToIndex(labels)
    return computeTestset(scalledData, labels, sizeOfTestset, mode)

#return scalled and prepared challenge dataset
def computeChallengeSet():
    data = extractionChallenge()
    scalledData = datasetPreparation(data)
    return scalledData
    
