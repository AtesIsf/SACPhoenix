import keras
from keras.layers import Dense, Dropout

### Creating the model ###

ARCHITECTURE = "3"

model = keras.Sequential()
model.add(keras.layers.Input(shape=(17,))) # Do not change the shape

# Tweak the layers in between
model.add(Dense(128))
model.add(Dropout(0.2))

model.add(Dense(64))
model.add(Dropout(0.2))

model.add(Dense(32))
model.add(Dropout(0.2))

model.add(Dense(16))
model.add(Dropout(0.2))


model.add(Dense(2)) # Do not change the shape