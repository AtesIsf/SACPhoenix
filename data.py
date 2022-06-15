from pandas import read_csv
import numpy as np

### Loading in the data ###
TEST_PERCENTAGE = 0.15
VAL_PERCENTAGE = 0.1

data = read_csv("./data_3.txt", header = None).to_numpy(copy=True)
np.random.shuffle(data)

val_end = int(len(data) * VAL_PERCENTAGE)
test_end = val_end + int(len(data) * TEST_PERCENTAGE)
valx = data[:val_end,:17]
valy = data[:val_end,17:]
testx = data[val_end:test_end,:17]
testy = data[val_end:test_end,17:]
trainx = data[test_end:,:17]
trainy = data[test_end:,17:]

print(f"Training: {len(trainx)} Validation: {len(valx)} Test: {len(testx)}")
input()