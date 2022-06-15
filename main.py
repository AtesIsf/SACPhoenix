from data import *
from model import *
from keras.callbacks import TensorBoard
import os

BATCH_SIZE = (256, 512)
EPOCHS = (200, 2000)
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

avg_errs = []

for b in BATCH_SIZE:
    for e in EPOCHS:
        name = f"Arch{ARCHITECTURE}-Batch{b}-Epoch{e}"
        tb = TensorBoard(os.path.join(CURRENT_DIR, "logs", name))

        model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
        model.fit(x=trainx, y=trainy, batch_size=b, epochs=e, validation_data=(valx, valy), callbacks=tb)

        model.save(os.path.join(CURRENT_DIR, "models", f"{name}.h5"))

        average_steering_error  = 0

        for i in range(len(testx)):
            predicted = model.predict(np.reshape(testx[i], (1,17)))
            steering_error = abs(testy[i][1] - predicted[0][1])
            average_steering_error += steering_error

            average_steering_error /= len(testx)

        print(f"Steering Error: {average_steering_error}")
        avg_errs.append(average_steering_error)
print(avg_errs)
