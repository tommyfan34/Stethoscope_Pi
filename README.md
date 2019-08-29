# Stethoscope_Pi
This is an electronic stethoscope using Raspberry Pi 3
Created by Xiao Fan @ UCLA

## Usage
```bash
cd stethoscope_pi
./samp.sh
```

## Codes
wavread.py: read the .wav file and get the data and sample rate
ALE_LMS.py: denoise the audio using adaptive line enhancer with least mean square (LMS)
ALE_NLMS.py: denoise the audio using adaptive line enhancer with normalized leasr mean square (NLMS)
ALE_RLS.py: denoise the audio using adaptive line enhancer with recursive least square (RLS)
none of these three denoising algorithm work well, so I didn't use them at last
