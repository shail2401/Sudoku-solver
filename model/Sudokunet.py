from tensorflow.keras.layers import Dropout, Conv2D, BatchNormalization, Dense, Flatten
from tensorflow.keras.models import Sequential


class SudokuNet:
    @staticmethod
    def build(height, width, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)
        model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(64, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(classes, activation='softmax'))

        return model
