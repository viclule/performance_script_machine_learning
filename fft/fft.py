# fft calculation functionality
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import time
import os


def calculate_fft(file_name, number_of_executions):
    rate, data = wav.read(_get_file_path(file_name))
    # start timing
    start = time.time()
    for _ in range(number_of_executions):
        try:
            __ = fft(data[:, 0])
        except IndexError:
            __ = fft(data)
    tm = time.time() - start
    size = len(data)
    return tm, rate, size


def _get_file_path(file_name):
    full_path = os.path.abspath(os.path.dirname('__file__')) + \
        '/fft/sound_files/' + file_name
    print(full_path)
    return full_path