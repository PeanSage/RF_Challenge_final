import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Flatten, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import logging

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomDataGenerator(Sequence):
    def __init__(self, data_file, batch_size=8, num_classes=9, shuffle=True):  # Update num_classes as appropriate
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.load_data()
        self.on_epoch_end()

    def load_data(self):
        data = np.fromfile(self.data_file, dtype=np.float32)
        num_samples = len(data) // (1024 * 2)  # Calculate number of complete signal sets
        remainder = len(data) % (1024 * 2)  # Calculate remainder to check for incomplete sets

        if remainder != 0:
            logging.warning(f"Data file contains incomplete signal set. Total data length: {len(data)}, expected multiple of {16384 * 2}. Truncating {remainder} values.")
            data = data[:-remainder]  # Truncate the incomplete set

        self.data = data.reshape(num_samples, 1024, 2)
        # Dummy labels for example, replace this with actual label extraction logic
        self.labels = np.random.randint(0, self.num_classes, num_samples)
        self.labels = to_categorical(self.labels, num_classes=self.num_classes)

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_X = self.data[start_idx:end_idx]
        batch_y = self.labels[start_idx:end_idx]
        return batch_X, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.data))
            np.random.shuffle(indices)
            self.data = self.data[indices]
            self.labels = self.labels[indices]

# Load class names from a text file
with open('rf_classes.txt', 'r') as file:
    next(file)  # Skip the first line
    class_names = file.read().splitlines()
    num_classes = len(class_names)

# Model definition. Sequential, Conv1D
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', padding='same', input_shape=(1024, 2)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Creating Data Generators
#play around with the batch size, recoomend keepig shuffle on. 
train_generator = CustomDataGenerator('training_data.dat', batch_size=8, shuffle=True)
test_generator = CustomDataGenerator('testing_data.dat', batch_size=4, shuffle=True)

# Train the model
model.fit(train_generator, validation_data=test_generator, epochs=1000)
#change the epochs when dealing with more or less data

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)

# Print evaluation metrics
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Save the trained model, 2 differet types for redundancy
model.save('RF_challenge_rev8.keras')
model.save('RF_challenge_rev8.h5')