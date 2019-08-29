# Stethoscope_Pi
This is an electronic stethoscope using Raspberry Pi 3
Created by Xiao Fan @ UCLA

## Prototype Setup


## Usage
In the Raspberry Pi, type in  
```bash
cd stethoscope_pi
./samp.sh
```
to start sampling the heart sound

## Codes
wavread.py: read the .wav file and get the data and sample rate  
ALE_LMS.py: denoise the audio using adaptive line enhancer with least mean square (LMS)  
ALE_NLMS.py: denoise the audio using adaptive line enhancer with normalized leasr mean square (NLMS)  
ALE_RLS.py: denoise the audio using adaptive line enhancer with recursive least square (RLS)  
none of these three denoising algorithm work well, so I didn't use them at last  
NASE.py: take the nomalized average shannon energy envelope of the heart sound  
HSSeg.py: 
