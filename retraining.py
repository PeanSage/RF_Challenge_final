from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Flatten, Dense
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback

class ClassificationCounter(Callback):
    def __init__(self, validation_generator, file_name):
        super(ClassificationCounter, self).__init__()
        self.validation_generator = validation_generator
        self.file_name = file_name
        # Write headers to CSV file
        with open(self.file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = ['Epoch'] + ['class_{}'.format(i) for i in range(len(validation_generator.label_encoder.classes_))]
            writer.writerow(headers)

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.validation_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        class_counts = np.bincount(predicted_classes, minlength=len(self.validation_generator.label_encoder.classes_))
        
        # Write classification counts to the CSV file
        with open(self.file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1] + class_counts.tolist())

class CSVDataGenerator(Sequence):
    def __init__(self, csv_file, batch_size=1, shuffle=True):
        self.df = pd.read_csv(csv_file)
        self.label_encoder = LabelEncoder()
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['Signal_Type'])
        self.batch_size = batch_size // 1024  # Number of signals per batch
        self.shuffle = shuffle
        self.n_classes = len(self.label_encoder.classes_)
        self.on_epoch_end()

    def __len__(self):
        return len(self.df) // (self.batch_size * 1024)

    def __getitem__(self, index):
        start_idx = index * self.batch_size * 1024
        end_idx = (index + 1) * self.batch_size * 1024
        batch_df = self.df.iloc[start_idx:end_idx]
        
        X = batch_df[['I', 'Q']].to_numpy().reshape(-1, 1024, 2)  # (-1 will automatically adjust based on the actual batch size)
        y = batch_df['label_encoded'].iloc[0::1024].to_numpy()
        y = to_categorical(y, num_classes=self.n_classes)

        return X, y

    def on_epoch_end(self):
        # Shuffle at the signal level, not individual rows
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)


# Load class names from a text file
with open('rf_classes.txt', 'r') as file:
    next(file)  # Skip the first line
    class_names = file.read().splitlines()
    num_classes = len(class_names)


# Load the previously trained model
model = load_model('RF_challenge_rev6.keras')

# Optional: Recompile the model if you need to make changes to the configuration
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Assuming you have a data generator already set up
train_generator = CSVDataGenerator('labeled_training_data.csv', batch_size=1024, shuffle=True)
test_generator = CSVDataGenerator('labeled_testing_data.csv', batch_size=1024, shuffle=True)

# Continue training
model.fit(train_generator, validation_data=test_generator, epochs=100)  # Adjust epochs or other parameters as needed

# Save the model again
model.save('RF_challenge_rev3_updated.keras')
model.save('RF_challenge_rev3_updated.h5')