from data import *
from model import *
from keras.callbacks import TensorBoard
import os

BATCH_SIZE = (32, 64, 128, 256)
EPOCHS = (100, 200)
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

for b in BATCH_SIZE:
    for e in EPOCHS:
        name = f"Arch{ARCHITECTURE}-Batch{b}-Epoch{e}"
        tb = TensorBoard(os.path.join(CURRENT_DIR, "logs", name))

        model.compile(optimizer="adam", loss="mse")#, metrics=["accuracy"])
        model.fit(x=trainx, y=trainy, batch_size=b, epochs=e, validation_data=(valx, valy), callbacks=tb)

        model.save(os.path.join(CURRENT_DIR, "models", f"{name}.h5"))