from data import *
from model import *
from keras.callbacks import TensorBoard

BATCH_SIZE = (32, 64, 128)
EPOCHS = (20, 40, 80)

for b in BATCH_SIZE:
    for e in EPOCHS:
        name = f"Batch:{b}-Epoch:{e}"
        tb = TensorBoard(f"./logs/{ARCHITECTURE}/{name}")

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x=trainx, y=trainy, batch_size=32, epochs=20, validation_data=(valx, valy), callbacks=tb)

        model.save(f"./models/{ARCHITECTURE}/{name}")