import model2 as model
import datasetPrep2 as data

m = model.getModel() 
trainX, trainY, testX, testY = data.computeDatasets()

def train(epochs=20):
    m.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    m.fit(trainX, trainY, epochs=epochs)
    m.save('models/modelTest')

train()
score = m.evaluate(testX, testY, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
