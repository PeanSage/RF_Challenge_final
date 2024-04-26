import tkinter as tk
from tkinter import messagebox
import numpy as np
import datetime
import threading
import subprocess
import csv
import tensorflow as tf
from tensorflow.keras.models import load_model
import geocoder
import subprocess
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Bidirectional

from tensorflow.keras.initializers import Orthogonal

#from main import main    #gnu radio function
import time

# Global flags and paths
sweep_flag = 0
class_names = []
is_paused = False
model_filepath = "RF_challenge_rev5.h5"
loaded_model = load_model(model_filepath)

dat_filepath = "freq_sweep_output.dat"
results_filepath = "predictions.csv"
class_filepath = "rf_classes.txt"
GNU_RADIO_SCRIPT = "main.py"

# GUI variables and main window initialization
main = tk.Tk()
main.geometry("500x500")
main.title("RF Challenge GUI")

var1 = tk.StringVar(value=[])  # Initialize with an empty list for class names
var2 = tk.StringVar(value=[])  # Initialize with an empty list for counts

# Static labels as headers for ListBoxes
header1 = tk.Label(main, text="Class", font=('Arial', 10))
header1.place(x=225, y=30)
header2 = tk.Label(main, text="Count", font=('Arial', 10))
header2.place(x=350, y=30)

def load_class_names():
    global class_names
    with open(class_filepath, 'r') as file:
        next(file)
        class_names = [line.strip() for line in file if line.strip()]



# Function to fetch geographic information and update GUI
def fetch_and_display_geo_info():
    try:
        g = geocoder.ip('me')
        lat, lng = g.latlng
        lat_lon_label.config(text=f"Latitude: {lat}, Longitude: {lng}")
    except Exception as e:
        print("Failed to get geo-info:", e)
        lat_lon_label.config(text="Latitude: N/A, Longitude: N/A")
    time_label.config(text=f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main.after(600000, fetch_and_display_geo_info)


def process_data():
    try:
        # Load the model
        loaded_model = load_model(model_filepath)

        # Read IQ data from file
        iq_data = np.fromfile(dat_filepath, dtype=np.float32)

        # Ensure the data contains complete blocks of 2048 floats (1024 I/Q pairs)
        full_blocks = len(iq_data) // 2048 * 2048
        iq_data = iq_data[:full_blocks]

        # Reshape the data to fit the model input shape
        if len(iq_data) > 0:
            iq_data = iq_data.reshape(-1, 1024, 2)
            predictions = loaded_model.predict(iq_data)
            predicted_classes = np.argmax(predictions, axis=1)
            class_counts = np.bincount(predicted_classes, minlength=len(class_names))
            # Write results to a CSV file
            with open(results_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Class', 'Count'])
                for i, count in enumerate(class_counts):
                    if count > 0:
                        writer.writerow([class_names[i], count])
            main.after(0, lambda: update_display_with_new_csv(results_filepath))
        else:
            print("No complete data blocks to process.")

    except Exception as e:
        print("Error during data processing:", e)


#function to display results to gui
def update_display_with_new_csv(results_filepath):
    try:
        with open(results_filepath, 'r') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip the header
            data = [row for row in reader if len(row) == 2]

        data.sort(key=lambda x: int(x[1]), reverse=True)
       
        # Extract class names and counts
        list_of_entries1 = [row[0] for row in data]
        list_of_entries2 = [row[1] for row in data]

        # Update the Listbox data and adjust height
        var1.set(list_of_entries1)
        var2.set(list_of_entries2)

        # Set Listbox height to the number of entries or a minimum value
        listbox_height = max(len(list_of_entries1), 1)  # Ensure there's at least 1 row
        listbox1.config(height=listbox_height)
        listbox2.config(height=listbox_height)

    except Exception as e:
        print("Error updating GUI:", e)

def start_button():
    global sweep_flag  # Ensure this is declared as global if you're changing its value
    if sweep_flag == 0:
        script_path = os.path.join(os.getcwd(), GNU_RADIO_SCRIPT)
        cmd = ["/usr/bin/python3", "-u", script_path]
        # Run GNU Radio and wait for it to finish
        subprocess.run(cmd, cwd=os.getcwd())
        sweep_flag = 1  # Set the flag to indicate that the sweep has been run

    # After GNU Radio has completed, process the data
    process_data()
    print("done processing data")


def pause_button():
    global is_paused
    is_paused = not is_paused
    btn_text2.set("Resume" if is_paused else "Pause")

    #possibly use pause button to make a new sweep and display said results.

def close_button():
    main.destroy()

# GUI Elements with styles
lat_lon_label = tk.Label(main, text="Latitude: , Longitude: ", font=('Arial', 10))
lat_lon_label.place(x=10, y=400)

time_label = tk.Label(main, text="Time: ", font=('Arial', 10))
time_label.place(x=10, y=425)

btn_text1 = tk.StringVar(value="Start")
B = tk.Button(main, textvariable=btn_text1, fg='white', bg='green', command=start_button)
B.place(x=15, y=50)
B.config(height=5, width=10)

btn_text2 = tk.StringVar(value="Pause")
C = tk.Button(main, textvariable=btn_text2, fg='black', bg='yellow', command=pause_button)
C.place(x=15, y=150)
C.config(height=5, width=10)

btn_text3 = tk.StringVar(value="Close")
D = tk.Button(main, textvariable=btn_text3, fg='white', bg='black', command=close_button)
D.place(x=15, y=250)
D.config(height=5, width=10)

listbox1 = tk.Listbox(main, listvariable=var1, height=10)
listbox1.place(x=225, y=50)
listbox2 = tk.Listbox(main, listvariable=var2, height=10)
listbox2.place(x=350, y=50)

fetch_and_display_geo_info()
load_class_names()
main.mainloop()