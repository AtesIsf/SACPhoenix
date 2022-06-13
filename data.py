from pandas import read_csv
import numpy as np

### Loading in the data ###
TEST_PERCENTAGE = 0.15
EVAL_PERCENTAGE = 0.15

data = read_csv("./data.txt", header = None).to_numpy(copy=True)
np.random.shuffle(data)

val_end = int(len(data) * EVAL_PERCENTAGE)
test_end = val_end + int(len(data) * TEST_PERCENTAGE)
valx = data[:val_end,:17]
valy = data[:val_end,17:]
testx = data[val_end:test_end,:17]
testy = data[val_end:test_end,17:]
trainx = data[test_end:,:17]
trainy = data[test_end:,17:]