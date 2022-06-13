import keras
from keras.layers import Dense, Dropout

### Creating the model ###

ARCHITECTURE = 1

model = keras.Sequential()
model.add(keras.layers.Input(shape=(17,))) # Do not change the shape

# Tweak the layers in between
model.add(Dense(200, activation="sigmoid"))
model.add(Dropout(0.2))

model.add(Dense(200, activation="sigmoid"))
model.add(Dropout(0.2))

model.add(Dense(100, activation="sigmoid"))
model.add(Dropout(0.2))

model.add(Dense(20, activation="sigmoid"))
model.add(Dropout(0.2))



model.add(Dense(2, activation="softmax")) # Do not change the shape