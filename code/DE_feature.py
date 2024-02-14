# %%
#from part1_preprocess import date2student_id
import tqdm
import numpy as np
from scipy.integrate import simps
import math
import os
import json

# %%
FREQ_BANDS = {    
    "delta": [0.5, 4],   
    "theta": [4, 8],     
    "alpha": [8, 13],    
    "beta": [13, 30],    
    "gamma": [30,50]
}

# %%
def bandpower(data, sf, band, method='welch', window_sec=None, relative=False):
    from scipy.signal import welch
    from mne.time_frequency import psd_array_multitaper

    band = np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        freqs, psd = welch(data, sf, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose=0)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

# %%
def get_bp(idx2eeg, out_path, video_duration=60, fs=1000, start_sec=0, end_sec=30):
    print('Extracting bp...', end=' ')
    idx2de = {}
    try:
        for idx in idx2eeg.keys():
            eeg = np.array(idx2eeg[idx])

            # Calculate the number of samples corresponding to the start and end times
            start_samples = start_sec * fs
            end_samples = end_sec * fs

            # Adjust the EEG data to only take data from 0-30 seconds of the video
            eeg = eeg[:, start_samples:end_samples]

            de_list = []

            # Calculate the number of times for feature extraction, ensuring not to exceed the length of the adjusted data
            num_of_extraction = min(int(eeg.shape[1] / 1000), end_sec - start_sec)

            for i in range(num_of_extraction):
                # Process 1000 samples each time
                tmp_data = eeg[:, i * 1000: i * 1000 + 1000]
                tmp_fs = []
                for channel_id in range(tmp_data.shape[0]):
                    tmp_feature = []
                    for band_item in FREQ_BANDS.values():
                        # Using de features
                        tmp_feature.append(math.log(bandpower(tmp_data[channel_id], fs, band_item,window_sec=1, relative=True)))
                    tmp_fs.append(tmp_feature)
                de_list.append(tmp_fs)
            idx2de[idx] = de_list

        # Use the with statement to ensure the file is closed properly
        with open(out_path, 'w') as file:
            json.dump(idx2de, file)

    except Exception as e:
        print("An error occurred: ", e)
        # Handle or log the exception, modify as needed

    print('Done.')

# %%
#add your participant list here
for date in tqdm.tqdm(['participant_1','participant_2'
]):
    idx2eeg = json.load(open('./x2eeg/'+date+'_idx2eeg.json'))
    get_bp(idx2eeg, './DE_features/'+date+'_idx2de.json')

# %%


# %%


# %%


# %%



