import tensorflow as tf

def getModel():
    model = tf.keras.Sequential(
        [ 
            tf.keras.Input(shape=(16)),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(22, activation=tf.nn.softmax)
        ]
    )
    
    return model
