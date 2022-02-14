import tensorflow as tf
import numpy as np
import pandas as pd
import datasetPrep


def getRes():
    challengeSet = datasetPrep.computeChallengeSet()
    genres = datasetPrep.getGenres()

    #model2 is the our best model
    model = tf.keras.models.load_model('models/model2')
    prediction = model.predict(challengeSet)
    
    #processing the results 
    results = []
    for res in prediction:
        index = np.argmax(res)
        results.append(genres[index])
    
    results = np.array(results)
    print(results)
    resDT = pd.DataFrame(results,columns=['genre'])    
    
    
    resDT.to_csv("output.csv", sep='\t', encoding='utf-8')

getRes()
