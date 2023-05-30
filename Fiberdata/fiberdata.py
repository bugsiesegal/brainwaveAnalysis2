import os
from datetime import timedelta
from typing import List

from matplotlib import pyplot as plt
from numpy import ndarray
import numpy as np
from scipy import stats
from scipy.signal import medfilt

from Fiberdata.tdt import TDTFile
import torch
from torch.utils.data import Dataset


class FiberData:
    data: ndarray
    fs: float
    name: str
    duration: timedelta
    channels: List[int]
    stream_channel: int
    ticks: ndarray
    offset: ndarray
    onset: ndarray

    def __init__(self, tdt_file: TDTFile, stream: str, stream_channel: int):
        self.data = tdt_file.get_streams()[stream].data[stream_channel]
        self.fs = tdt_file.get_streams()[stream].fs
        self.name = tdt_file.get_filename()
        self.duration = tdt_file.get_info().duration
        self.channels = tdt_file.get_streams()[stream].channel
        self.stream_channel = stream_channel
        self.ticks = tdt_file.get_epocs()['Tick'].data
        self.offset = tdt_file.get_epocs()['Tick'].offset
        self.onset = tdt_file.get_epocs()['Tick'].onset

    def get_time_vector(self) -> np.ndarray:
        """
        Generate a time vector for the data.

        Returns:
        -------
        time_vector : np.ndarray
            A time vector corresponding to the data's timestamps.
        """
        time_vector = np.arange(0, len(self.data)) / self.fs
        return time_vector

    def get_channel_data(self, channel: int) -> np.ndarray:
        """
        Retrieve data for a specific channel.

        Parameters:
        ----------
        channel : int
            The channel number to retrieve data for.

        Returns:
        -------
        channel_data : np.ndarray
            The data for the specified channel.
        """
        if channel not in self.channels:
            raise ValueError(f"Channel {channel} not available in this dataset.")
        channel_data = self.data[channel]
        return channel_data

    def resample_data(self, new_fs: float) -> None:
        """
        Resample the data to a new sampling rate.

        Parameters:
        ----------
        new_fs : float
            The new sampling rate (in Hz) to resample the data to.
        """
        from scipy.signal import resample

        num_samples = int(len(self.data) * new_fs / self.fs)
        resampled_data = resample(self.data, num_samples)
        self.data = resampled_data
        self.fs = new_fs

    def filter_data(self, lowcut: float, highcut: float, order: int = 4) -> None:
        """
        Apply a bandpass filter to the data.

        Parameters:
        ----------
        lowcut : float
            The lower frequency cutoff (in Hz) for the filter.
        highcut : float
            The upper frequency cutoff (in Hz) for the filter.
        order : int, optional, default: 4
            The order of the filter.
        """
        from scipy.signal import butter, filtfilt

        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, self.data)
        self.data = filtered_data

    def plot_data(self, start_time: float = None, end_time: float = None) -> None:
        """
        Plot the data.

        Parameters:
        ----------
        start_time : float, optional
            The starting time (in seconds) for the plot. If None, start from the beginning.
        end_time : float, optional
            The ending time (in seconds) for the plot. If None, plot until the end.
        """
        import matplotlib.pyplot as plt

        time_vector = self.get_time_vector()

        if start_time is not None:
            start_index = int(start_time * self.fs)
            time_vector = time_vector[start_index:]
            data = self.data[start_index:]
        else:
            data = self.data

        if end_time is not None:
            end_index = int(end_time * self.fs)
            time_vector = time_vector[:end_index]
            data = data[:end_index]

        plt.plot(time_vector, data)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Fiber Data: {self.name} - Channel {self.stream_channel}')
        plt.show()

    def plot_fft(self, window_size: int = 50, threshold: float = 2.0) -> None:
        """
        Plot the Fourier Transform of the data using a log scale for the amplitude,
        and mark outliers based on a local average of the log-transformed data.

        Parameters:
        ----------
        window_size : int, optional, default: 50
            The size of the rolling window for calculating the local average.
        threshold : float, optional, default: 2.0
            The threshold to identify outliers based on deviations from the local average.
        """
        # Calculate the Fast Fourier Transform (FFT) of the data
        fft_data = np.fft.fft(self.data)
        fft_freqs = np.fft.fftfreq(len(self.data), 1 / self.fs)

        # Calculate the power spectral density (PSD) in dB
        psd_data = 20 * np.log10(np.abs(fft_data))

        # Calculate the local mean and standard deviation using a rolling window on the log-transformed data
        local_mean = np.convolve(psd_data, np.ones(window_size) / window_size, mode='valid')
        local_std = np.array([np.std(psd_data[i:i + window_size]) for i in range(len(psd_data) - window_size + 1)])

        # Identify the outliers based on deviations from the local average of the log-transformed data
        deviations = (psd_data[window_size - 1:] - local_mean) / local_std
        outliers = np.where(np.abs(deviations) > threshold)[0] + window_size // 2

        # Plot the PSD
        plt.figure()
        plt.plot(fft_freqs, psd_data, label='PSD')
        plt.scatter(fft_freqs[outliers], psd_data[outliers], color='red', marker='o', label='Outliers')

        # Customize the plot
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.title(f'Fourier Transform (Log Scale) with Outliers: {self.name} - Channel {self.stream_channel}')
        plt.legend()
        plt.show()

        # Print the frequency of the outliers
        outlier_freqs = fft_freqs[outliers]
        print("Outlier frequencies: ", outlier_freqs)

    def apply_median_filter(self, kernel_size: int = 3) -> None:
        """
        Apply a median filter to the data to smooth out large spikes.

        Parameters:
        ----------
        kernel_size : int, optional, default: 3
            The size of the kernel for the median filter. Must be odd.
        """
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        self.data = medfilt(self.data, kernel_size)

    def normalize_data(self, method: str = 'minmax', remove_spikes: bool = True, kernel_size: int = 3) -> None:
        """
        Normalize the data.

        Parameters:
        ----------
        method : str, optional, default: 'minmax'
            The normalization method to use. Options: 'minmax' or 'zscore'.
        remove_spikes : bool, optional, default: True
            Whether to apply a median filter to remove large spikes before normalization.
        kernel_size : int, optional, default: 3
            The size of the kernel for the median filter (if applied). Must be odd.
        """
        if remove_spikes:
            self.apply_median_filter(kernel_size)

        if method == 'minmax':
            min_value = np.min(self.data)
            max_value = np.max(self.data)
            self.data = (self.data - min_value) / (max_value - min_value)
        elif method == 'zscore':
            mean = np.mean(self.data)
            std_dev = np.std(self.data)
            self.data = (self.data - mean) / std_dev
        else:
            raise ValueError("Invalid normalization method. Use 'minmax' or 'zscore'.")

    def save_data(self, directory: str = "saved_data", filename: str = None) -> None:
        """
        Save the fiber data and its attributes to a file.

        Parameters:
        ----------
        directory : str, optional, default: "saved_data"
            The directory to save the file in. If it does not exist, it will be created.
        filename : str, optional
            The name of the file to save the data in. If not provided, use the original TDT filename.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        if filename is None:
            filename = self.name

        filepath = os.path.join(directory, f"{filename}.npy")

        attributes = {
            "data": self.data,
            "fs": self.fs,
            "name": self.name,
            "duration": self.duration,
            "channels": self.channels,
            "stream_channel": self.stream_channel,
            "ticks": self.ticks,
            "offset": self.offset,
            "onset": self.onset,
        }

        np.save(filepath, attributes)
        print(f"Data and attributes saved to: {filepath}")

    @classmethod
    def from_saved_data(cls, filepath: str) -> 'FiberData':
        """
        Load fiber data and its attributes from a file and create a FiberData object.

        Parameters:
        ----------
        filepath : str
            The path to the file containing the saved data and attributes.

        Returns:
        -------
        fiber_data : FiberData
            A FiberData object containing the loaded data and attributes.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        attributes = np.load(filepath, allow_pickle=True).item()

        fiber_data = cls.__new__(cls)
        fiber_data.data = attributes["data"]
        fiber_data.fs = attributes["fs"]
        fiber_data.name = attributes["name"]
        fiber_data.duration = attributes["duration"]
        fiber_data.channels = attributes["channels"]
        fiber_data.stream_channel = attributes["stream_channel"]
        fiber_data.ticks = attributes["ticks"]
        fiber_data.offset = attributes["offset"]
        fiber_data.onset = attributes["onset"]

        return fiber_data

    def to_csv(self, path):
        """
        Save the data to a csv file.

        Parameters:
        ----------
        path : str
            The path to save the file to.
        """
        np.savetxt(path, self.data, delimiter=",")


class SlidingWindowDataset(Dataset):
    def __init__(self, fiber_data: FiberData, window_size: int, step_size: int):
        self.fiber_data = fiber_data
        self.window_size = window_size
        self.step_size = step_size
        self.num_windows = (len(self.fiber_data.data) - self.window_size) // self.step_size + 1

        self.precompile_windows()

    def precompile_windows(self) -> None:
        self.windows = []

        for idx in range(self.num_windows):
            start_index = idx * self.step_size
            end_index = start_index + self.window_size

            window_data = self.fiber_data.data[start_index:end_index]
            window_data = self.normalize_window(window_data)
            self.windows.append(window_data)

    def normalize_window(self, window_data: np.ndarray) -> np.ndarray:
        min_value = np.min(window_data)
        max_value = np.max(window_data)
        normalized_data = (window_data - min_value) / (max_value - min_value)

        return normalized_data

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.num_windows:
            raise IndexError(f"Index {idx} is out of bounds for SlidingWindowDataset of size {self.num_windows}")

        window_data = self.windows[idx]
        return torch.from_numpy(window_data).unsqueeze(0)  # Add an extra dimension


