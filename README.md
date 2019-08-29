# Stethoscope_Pi
This is an electronic stethoscope using Raspberry Pi 3
Created by Xiao Fan @ UCLA Jul 2019  
Email: tommyfanzju@gmail.com

## Prototype Setup
![image of prototype setup](https://github.com/tommyfan34/Stethoscope_Pi/blob/master/Documents/WeChat%20Screenshot_20190829084743.png)

## Usage
Press the button on the battery holder to turn on the power  
In the Raspberry Pi, type in  
```bash
cd stethoscope_pi
./samp.sh
```
to start sampling the heart sound

## Codes
* wavread.py: read the .wav file and get the data and sample rate  
* ALE_LMS.py: denoise the audio using adaptive line enhancer with least mean square (LMS)  
* ALE_NLMS.py: denoise the audio using adaptive line enhancer with normalized leasr mean square (NLMS)  
* ALE_RLS.py: denoise the audio using adaptive line enhancer with recursive least square (RLS)  
none of these three denoising algorithm work well, so I didn't use them at last  
* NASE.py: take the nomalized average shannon energy envelope of the heart sound  
* HSSeg.py: segment the heart sound into 4 periods: S1, systole, S2, diastole  
* feature_extraction.py: extract the features in time domain and frequency domain (including Mel Frequncy Cepstral Coefficients).  
The .wav files whose features are to be extracted are specified in training_file.xlsx, with the first column to be the original file name, second column is the label (0 is normal, 1 is abnormal), third column is the diagnosis, fourth column is the modified file name. feature.csv is comprised of the feature vector extracted. For each sample, 120 features are extracted. label.csv is the label of each sample.  
* model_training.py: 
