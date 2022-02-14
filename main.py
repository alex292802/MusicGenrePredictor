import model 
import datasetPrep

m = model.getModel() 
trainX, trainY, testX, testY = datasetPrep.computeDatasets()

def train(epochs=50):
    m.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    m.fit(trainX, trainY, epochs=epochs)
    m.save('models/modelTest')

train()
score = m.evaluate(testX, testY, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
