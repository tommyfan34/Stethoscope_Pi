#!/bin/bash
arecord -vv -f S16_LE -c1 -d5 -r4800 /home/pi/stethoscope_pi/test.wav
python3 -W ignore HSSeg.py
python3 -W ignore model_test.py
