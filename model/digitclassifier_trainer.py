import argparse
import Sudokunet
from tensorflow.keras.datasets import mnist
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to output model after training")
args = vars(ap.parse_args())

INIT_LR = 1e-3
EPOCHS = 30
BS = 128
best_weight = ModelCheckpoint(args['model'], monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, min_lr=1e-6)
CALLBACKS = [best_weight, reduce_lr]

print("[INFO] accessing data...")
((train_X, train_y), (test_X, test_y)) = mnist.load_data()
# add a channel (i.e., grayscale) dimension to the digits
train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))
# scale data to the range of [0, 1]
train_X = train_X.astype("float32") / 255.0
test_X = test_X.astype("float32") / 255.0
# convert the labels from integers to vectors
le = LabelBinarizer()
train_y = le.fit_transform(train_y)
test_y = le.transform(test_y)

# Shuffling and Splitting Data
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

# Compiling Model
model = Sudokunet.SudokuNet.build(28, 28, 1, 10)
opt = RMSprop(lr=INIT_LR, rho=0.9, epsilon = 1e-08, decay=0.0)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
print(train_X.shape)
# Training Model
history = model.fit(train_X, train_y, batch_size=BS, epochs = EPOCHS, validation_data = (valid_X, valid_y), verbose = 1, callbacks=CALLBACKS)

# Evaluating Model
print("[INFO] evaluating network...")
predictions = model.predict(test_X)
print(classification_report(
	test_y.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in le.classes_]))
