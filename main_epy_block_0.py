import numpy as np
from gnuradio import gr

class MyPythonBlock(gr.sync_block):
    def __init__(self, center_freq_start=100e6, center_freq_stop=200e6, sample_rate=2.4e6, num_samples=100):
        gr.sync_block.__init__(
            self,
            name='MyPythonBlock',
            in_sig=[np.complex64],  # Input signal
            out_sig=[(np.float32, 1), np.complex64]  # Output signals (one for file sink and one for GUI sink)
        )

        # Define parameters
        self.center_freq_start = center_freq_start
        self.center_freq_stop = center_freq_stop
        self.sample_rate = sample_rate
        self.num_samples = num_samples  # Added this line

        # Initialize state variables
        self.current_frequency = center_freq_start
        self.is_recording = False
        self.samples_collected = 0  # counter for collected samples

        # MPSK SNR Estimator parameters
        self.snr_threshold = 10  # Adjust as needed

        # Flag to indicate if end of frequency range has been reached
        self.end_of_range_reached = False

    def work(self, input_items, output_items):
        # Check if end of frequency range has been reached
        if self.end_of_range_reached:
            # Stop receiving data
            return 0

        # Perform MPSK SNR estimation
        signal_power = np.mean(np.abs(input_items[0])**2)
        noise_power = np.var(input_items[0])
        snr = 10 * np.log10(signal_power / noise_power)

        # Check if SNR is above the threshold and no sample has been recorded yet
        if snr > self.snr_threshold and not self.is_recording:
            self.is_recording = True
            print(f"Recording started at frequency: {self.current_frequency}")

            # Code to start recording goes here
            # Placeholder: Replace this with actual recording code
            # Reset samples counter
            self.samples_collected = 0

        # Check if SNR drops below the threshold and recording is in progress
        elif snr <= self.snr_threshold and self.is_recording:
            self.is_recording = False
            print(f"Recording stopped at frequency: {self.current_frequency}")

            # Code to stop recording goes here
            # Placeholder: Replace this with actual recording code

        # Check if recording is in progress and not enough samples collected
        if self.is_recording and self.samples_collected < self.num_samples:
            # Increment samples counter
            self.samples_collected += len(input_items[0])

        # Move to the next frequency
        self.current_frequency += self.sample_rate

        # Check if reached the end of frequency range
        if self.current_frequency > self.center_freq_stop:
            if not self.end_of_range_reached:
                print("End of frequency range reached.")
                self.end_of_range_reached = True

        # Copy input to first output (casting to float32)
        output_items[0][:] = input_items[0].real.astype(np.float32)

        # Copy input to second output
        output_items[1][:] = input_items[0]

        # Return the number of samples consumed (same as the input size)
        return len(input_items[0])

