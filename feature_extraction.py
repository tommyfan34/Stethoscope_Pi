from scipy.stats import skew, kurtosis
from wavread import wavread
from matplotlib.pyplot import *
from HSSeg import HSSeg
from NASE import NASE
from MFCC import MFCC

"""
This function is to extract the feature in time domain & frequency domain
Time domain features: 
1. Heart cycle intervals
2. S1, systole, S2, diastole intervals
3. ratio of systolic interval to RR interval of each heart cycle
4. ratio of diastolic interval to RR interval of each heart cycle
5. ratio of systolic to diastolic interval of each heart cycle
6. ratio of the mean absolute amplitude during systole to that during the S1 period in each heart cycle
7. ratio of the mean absolute amplitude during diastole to that during the S2 period in each heart cycle
8. skewness and kurtosis of the amplitude during S1, Systole, S2 and Diastole

Frequency domain features:
1. The power density spectrum of S1, Systole, S2, Diastole across 9 frequency bands: 
25-45, 45-65, 65-85, 85-105, 105-125, 125-150, 150-200, 200-300, 300-400 Hz
2. 12 Mel Frequency Cepstral Coefficients for each of S1, Systole, S2 and Diastole phases of the heart sound

Author: Xiao Fan
Date: 8/23/2019
Reference: https://github.com/Gvith/Heart-Sound-Classification/blob/master/feature_extraction.py
"""

NFFT = 256
frequency_ranges = [(25, 45), (45, 65), (65, 85), (85, 105), (105, 125), (125, 150), (150, 200), (200, 300),
                    (300, 500)]
freq = np.array([i / int(NFFT / 2) * 500 for i in range(0, int(NFFT / 2))])
f_indices = []
for each_freq_band in frequency_ranges:
    index = np.where(np.logical_and(freq >= each_freq_band[0], freq < each_freq_band[1]))
    f_indices.append(index)



"""
time_feature function is to extract the 
1. duration
2. mean value
3. skewness
4. kurtosis
of the sequence
"""

def time_feature(signal, samplerate):
    interval_length = np.around(len(signal)/samplerate, decimals=4)
    mean_value = np.mean(np.around(np.absolute(signal), decimals=4))
    seq_skew = skew(signal)
    seq_kurtosis = kurtosis(signal)
    return interval_length, mean_value, seq_skew, seq_kurtosis

"""
frequency_feature is to extract 
the power density spectrum across 9 frequency bands: 
25-45, 45-65, 65-85, 85-105, 105-125, 125-150, 150-200, 200-300, 300-400 Hz
"""

def frequncy_feature(signal, samplerate):

    # computes the power spectrum of the signal
    hamming_distance = signal * np.hamming(len(signal))
    fft = np.absolute(np.fft.rfft(hamming_distance, NFFT))
    power_spec = np.around(fft[:NFFT / 2], decimals=4)
    p_spec = ((1.0 / NFFT) * ((fft) ** 2))

"""
feature_extraction is to extract the feature from a HS record
"""
def feature_extraction(signal,samplerate,s1_start,s1_end,s2_start,s2_end):
    skew_list = [[],[],[],[]]
    kurtosis_list = [[],[],[],[]]
    interval_len_list = [[],[],[],[]]
    ratio_sys_rr = []
    ratio_dia_rr = []
    ratio_sys_dia = []
    ratio_sys_s1 = []
    ratio_dia_s2 = []
    mel_list = []
    power_list = []
    power_freq = []
    rr_list = []
    feature_vector = []
    for s1_index, s1_y in enumerate(s1_start):
        if s1_index == len(s1_start)-1:
            break
        # find the min # that's bigger than s1_start[s1_index] in s1_end
        for i, yi in enumerate(s1_end):
            if s1_end[i] > s1_start[s1_index]:
                break
        min_index = i
        # calculate the feature of s1
        interval_length_s1, mean_s1, seq_skew_s1, seq_kurtosis_s1 \
            = time_feature(signal[s1_start[s1_index]:s1_end[min_index]], samplerate)
        interval_len_list[0].append(interval_length_s1)
        skew_list[0].append(seq_skew_s1)
        kurtosis_list[0].append(seq_kurtosis_s1)

        # find the min # that's bigger than s1_end[min_index] in s2_start
        for i, yi in enumerate(s2_start):
            if s2_start[i] > s1_end[min_index]:
                break
        # calculate the feature of systole
        interval_length_sys, mean_sys, seq_skew_sys, seq_kurtosis_sys \
            = time_feature(signal[s1_end[min_index]:s2_start[i]], samplerate)
        interval_len_list[1].append(interval_length_sys)
        skew_list[1].append(seq_skew_sys)
        kurtosis_list[1].append(seq_kurtosis_sys)
        min_index = i

        # find the min # that's bigger than s2_start[min_index] in s2_end
        for i, yi in enumerate(s2_end):
            if s2_end[i] > s2_start[min_index]:
                break
        # calculate the feature of s2
        interval_length_s2, mean_s2, seq_skew_s2, seq_kurtosis_s2 \
            = time_feature(signal[s2_start[min_index]:s2_end[i]], samplerate)
        interval_len_list[2].append(interval_length_s2)
        skew_list[2].append(seq_skew_s2)
        kurtosis_list[2].append(seq_kurtosis_s2)
        min_index = i

        # find the min # that's bigger than s2_end[min_index] in s1_start
        for i, yi in enumerate(s1_start):
            if s1_start[i] > s2_end[min_index]:
                break
        # calculate the feature of diastole
        interval_length_dia, mean_dia, seq_skew_dia, seq_kurtosis_dia \
            = time_feature(signal[s2_end[min_index]:s1_start[i]], samplerate)
        interval_len_list[3].append(interval_length_dia)
        skew_list[3].append(seq_skew_dia)
        kurtosis_list[3].append(seq_kurtosis_dia)
        # calculate the duration of a heart cycle
        interval_length_rr = interval_length_s1 + interval_length_s2 \
            + interval_length_sys + interval_length_dia
        rr_list.append(interval_length_rr)
        # calculate the ratio
        ratio_sys_rr.append(interval_length_sys/float(interval_length_rr))
        ratio_dia_rr.append(interval_length_dia/float(interval_length_rr))
        ratio_sys_dia.append(interval_length_sys/float(interval_length_dia))
        ratio_sys_s1.append(mean_sys/float(mean_s1))
        ratio_dia_s2.append(mean_dia/float(mean_s2))

    # calculate the mean and standard deviation of the features in all periods
    mean_RR = np.around(np.mean(rr_list), decimals=4)
    std_RR = np.around(np.std(rr_list), decimals=4)
    mean_interval_s1 = np.around(np.mean(interval_len_list[0]), decimals=4)
    std_interval_s1 = np.around(np.std(interval_len_list[0]), decimals=4)
    mean_interval_sys = np.around(np.mean(interval_len_list[1]), decimals=4)
    std_interval_sys = np.around(np.std(interval_len_list[1]), decimals=4)
    mean_interval_s2 = np.around(np.mean(interval_len_list[2]), decimals=4)
    std_interval_s2 = np.around(np.std(interval_len_list[2]), decimals=4)
    mean_interval_dia = np.around(np.mean(interval_len_list[3]), decimals=4)
    std_interval_dia = np.around(np.std(interval_len_list[3]), decimals=4)
    mean_ratio_sys_rr = np.around(np.mean(ratio_sys_rr), decimals=4)
    std_ratio_sys_rr = np.around(np.std(ratio_sys_rr), decimals=4)
    mean_ratio_dia_rr = np.around(np.mean(ratio_dia_rr), decimals=4)
    std_ratio_dia_rr = np.around(np.std(ratio_dia_rr), decimals=4)
    mean_ratio_sys_dia = np.around(np.mean(ratio_sys_dia), decimals=4)
    std_ratio_sys_dia = np.around(np.std(ratio_sys_dia), decimals=4)
    mean_ratio_sys_s1 = np.around(np.mean(ratio_sys_s1), decimals=4)
    std_ratio_sys_s1 = np.around(np.std(ratio_sys_s1), decimals=4)
    mean_ratio_dia_s2 = np.around(np.mean(ratio_dia_s2), decimals=4)
    std_ratio_dia_s2 = np.around(np.std(ratio_dia_s2), decimals=4)
    mean_s1_skew = np.around(np.mean(skew_list[0]), decimals=4)
    std_s1_skew = np.around(np.std(skew_list[0]), decimals=4)
    mean_systole_skew = np.around(np.mean(skew_list[1]), decimals=4)
    std_systole_skew = np.around(np.std(skew_list[1]), decimals=4)
    mean_s2_skew = np.around(np.mean(skew_list[2]), decimals=4)
    std_s2_skew = np.around(np.std(skew_list[2]), decimals=4)
    mean_diastole_skew = np.around(np.mean(skew_list[3]), decimals=4)
    std_diastole_skew = np.around(np.std(skew_list[3]), decimals=4)
    mean_s1_kurtosis = np.around(np.mean(kurtosis_list[0]), decimals=4)
    std_s1_kurtosis = np.around(np.std(kurtosis_list[0]), decimals=4)
    mean_systole_kurtosis = np.around(np.mean(kurtosis_list[1]), decimals=4)
    std_systole_kurtosis = np.around(np.std(kurtosis_list[1]), decimals=4)
    mean_s2_kurtosis = np.around(np.mean(kurtosis_list[2]), decimals=4)
    std_s2_kurtosis = np.around(np.std(kurtosis_list[2]), decimals=4)
    mean_diastole_kurtosis = np.around(np.mean(kurtosis_list[3]), decimals=4)
    std_diastole_kurtosis = np.around(np.std(kurtosis_list[3]), decimals=4)

    feature_vector = [mean_RR, std_RR, mean_interval_s1, std_interval_s1, mean_interval_sys, std_interval_sys, \
                      mean_interval_s2, std_interval_s2, mean_interval_dia, std_interval_dia, mean_ratio_sys_rr, \
                      std_ratio_sys_rr, mean_ratio_dia_rr, std_ratio_dia_rr, mean_ratio_sys_dia, std_ratio_sys_dia, \
                      mean_ratio_sys_s1, std_ratio_sys_s1, mean_ratio_dia_s2, std_ratio_dia_s2, mean_s1_skew, \
                      std_s1_skew, mean_systole_skew, std_systole_skew, mean_s2_skew, std_s2_skew, mean_diastole_skew, \
                      std_diastole_skew, mean_s1_kurtosis, std_s1_kurtosis, mean_systole_kurtosis, std_systole_kurtosis, \
                      mean_s2_kurtosis, std_s2_kurtosis, mean_diastole_kurtosis, std_diastole_kurtosis]
    return feature_vector

if __name__ == "__main__":
    path = ["test.wav"]
    audio_clip = [1,5]
    for i, yi in enumerate(path):
        wavdata, wavtime, samplerate = wavread(yi, audio_clip)
        wavdata2, wavtime2 = NASE(wavdata, 0.02 * samplerate, samplerate, audio_clip[0])
        peak, s1, s1_start, s1_end, s2, s2_start, s2_end = HSSeg(wavdata2, wavtime2)
        s1_start = ((s1_start-audio_clip[0])*samplerate).astype(int)
        s1_end = ((s1_end-audio_clip[0])*samplerate).astype(int)
        s2_start = ((s2_start-audio_clip[0])*samplerate).astype(int)
        s2_end = ((s2_end-audio_clip[0])*samplerate).astype(int)
        feature_vector = feature_extraction(wavdata, samplerate, s1_start, s1_end, s2_start, s2_end)

