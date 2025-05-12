import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


class MNISTClassifier:
    def __init__(self):
        self.model = None

    def build_model(self):
        self.model = Sequential([
            # sequential model is a linear stack of layers - each layer has exactly one input and one output
            Conv2D(32, # number of kernels, each corresponds to one neuron
                   kernel_size=3,
                   activation='relu',
                   input_shape=(28, 28, 1)),
            BatchNormalization(),
            MaxPooling2D(pool_size=2), # pooling reduces dimensions, max pooling takes the max value in the pool
            # less dimensions == less computation, less overfitting
            Conv2D(64,
                   kernel_size=3,
                   activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=2),
            Flatten(), # flatten the 2D matrix to 1D vector
            Dense(128, # empirically chosen number for best performance, normal hyperparameter
                  activation='relu'), # fully connected layer, each neuron is connected to all neurons in the previous layer
            Dropout(0.5), # percentage of neurons tuning off randomly during training to prevent overfitting
            Dense(10, # number of classes/labels
                  activation='softmax')
        ])
        self.model.compile(optimizer='adam', # popular method - alternatives are for ex.: SGD, RMSprop, Adagrad
                           # loss function for multi-class classification
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
