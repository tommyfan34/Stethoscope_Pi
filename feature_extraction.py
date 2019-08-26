from scipy.stats import skew, kurtosis
from wavread import wavread
from matplotlib.pyplot import *
from HSSeg import HSSeg
from NASE import NASE
from scipy.fftpack import dct
import csv
import xlrd

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

# computes the mel coefficients for the sound signal.
# The theoretical underpinnings are as explained in http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#computing-the-mel-filterbank
def mel_coefficients(signal, sample_rate, nfilt):
    hamming_distance = signal * np.hamming(len(signal))
    fft = np.absolute(np.fft.rfft(hamming_distance, NFFT))
    pow_frames = ((1.0 / NFFT) * ((fft) ** 2))
    low_freq_mel = 0
    num_mel_coeff = 12
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2.0) / 700.0))  # Convert Hz to Mel

    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz

    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    mfcc = dct(filter_banks, type=2, axis=0, norm='ortho')[1:(num_mel_coeff+1)]
    (ncoeff,) = mfcc.shape
    cep_lifter = ncoeff
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    return mfcc

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
    power_spec = np.around(fft[:int(NFFT/2)], decimals=4)
    median_power = []
    for r, each_f_index in enumerate(f_indices):
        median_power.append(power_spec[each_f_index])
    return median_power

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
    mel_list = [[],[],[],[]]
    power_list = [[],[],[],[]]
    power_freq = []
    power_spectrum_s1 = []
    power_spectrum_sys = []
    power_spectrum_s2 = []
    power_spectrum_dia = []
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
        # mel_list[0].append(MFCC(signal[s1_start[s1_index]:s1_end[min_index]], samplerate))
        mel_list[0].append(mel_coefficients(signal[s1_start[s1_index]:s1_end[min_index]], samplerate, 40))
        power_list[0].append(frequncy_feature(signal[s1_start[s1_index]:s1_end[min_index]], samplerate))

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
        # mel_list[1].append(MFCC(signal[s1_end[min_index]:s2_start[i]], samplerate))
        mel_list[1].append(mel_coefficients(signal[s1_end[min_index]:s2_start[i]], samplerate, 40))
        power_list[1].append(frequncy_feature(signal[s1_end[min_index]:s2_start[i]], samplerate))
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
        # mel_list[2].append(MFCC(signal[s2_start[min_index]:s2_end[i]], samplerate))
        mel_list[2].append(mel_coefficients(signal[s2_start[min_index]:s2_end[i]], samplerate, 40))
        power_list[2].append(frequncy_feature(signal[s2_start[min_index]:s2_end[i]], samplerate))
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
        # mel_list[3].append(MFCC(signal[s2_end[min_index]:s1_start[i]], samplerate))
        mel_list[3].append(mel_coefficients(signal[s2_end[min_index]:s1_start[i]], samplerate, 40))
        power_list[3].append(frequncy_feature(signal[s2_end[min_index]:s1_start[i]], samplerate))
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
    # compute the first 12 mel frequency cepstral coefficients for each of the 4 different heart sound state
    mfcc_s1 = list(np.around(np.median(mel_list[0], axis=0), decimals=4))
    mfcc_systole = list(np.around(np.median(mel_list[1], axis=0), decimals=4))
    mfcc_s2 = list(np.around(np.median(mel_list[2], axis=0), decimals=4))
    mfcc_diastole = list(np.around(np.median(mel_list[3], axis=0), decimals=4))
    # compute the power spectrum of different periods
    for t in range(len(f_indices)):
        temp = []
        for i in range(len(power_list[0])):
            temp.append(np.median(power_list[0][i][t]))
        power_spectrum_s1.append(np.around(np.median(temp), decimals=4))
        temp = []
        for i in range(len(power_list[1])):
            temp.append(np.median(power_list[1][i][t]))
        power_spectrum_sys.append(np.around(np.median(temp), decimals=4))
        temp = []
        for i in range(len(power_list[2])):
            temp.append(np.median(power_list[2][i][t]))
        power_spectrum_s2.append(np.around(np.median(temp), decimals=4))
        temp = []
        for i in range(len(power_list[3])):
            temp.append(np.median(power_list[3][i][t]))
        power_spectrum_dia.append(np.around(np.median(temp), decimals=4))

    feature_vector = [mean_RR, std_RR, mean_interval_s1, std_interval_s1, mean_interval_sys, std_interval_sys, \
                      mean_interval_s2, std_interval_s2, mean_interval_dia, std_interval_dia, mean_ratio_sys_rr, \
                      std_ratio_sys_rr, mean_ratio_dia_rr, std_ratio_dia_rr, mean_ratio_sys_dia, std_ratio_sys_dia, \
                      mean_ratio_sys_s1, std_ratio_sys_s1, mean_ratio_dia_s2, std_ratio_dia_s2, mean_s1_skew, \
                      std_s1_skew, mean_systole_skew, std_systole_skew, mean_s2_skew, std_s2_skew, mean_diastole_skew, \
                      std_diastole_skew, mean_s1_kurtosis, std_s1_kurtosis, mean_systole_kurtosis, std_systole_kurtosis, \
                      mean_s2_kurtosis, std_s2_kurtosis, mean_diastole_kurtosis, std_diastole_kurtosis]
    feature_vector.extend(mfcc_s1)
    feature_vector.extend(mfcc_systole)
    feature_vector.extend(mfcc_s2)
    feature_vector.extend(mfcc_diastole)
    feature_vector.extend(power_spectrum_s1)
    feature_vector.extend(power_spectrum_sys)
    feature_vector.extend(power_spectrum_s2)
    feature_vector.extend(power_spectrum_dia)
    return feature_vector

if __name__ == "__main__":
    # read the xlsx file that specify the audio file
    data = xlrd.open_workbook("training_file.xlsx")
    table = data.sheets()[0]
    path = table.col_values(3)

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
        # write the feature vector into csv file
        with open("feature.csv", "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(feature_vector)
        # HS type: 0 is normal, 1 is abnormal
        HS_type = path[i][0]
        with open("label.csv", "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([HS_type])



