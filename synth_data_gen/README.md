# Steps to generate synthetic 360-degree audiovisual scapes

## 1 - Download the METU Sparg dataset

You can find the dataset at [https://zenodo.org/records/2635758](https://zenodo.org/records/2635758).

## 2 - Process the EM32 IRs

The RIRs are given as separate wavefiles for each of the 32 channels in the mic array. We need to join them into a single wavefile. In the codebase we use, each location has a wavefile, we call the joined wavfile as `IR_em32.wav`. Use the `remix_metu_rirs.py` script we provide.

## 3 - Download the video assets or collect them yourself from YouTube or other video libraries

## 4 - Execute audiovisual synthetic data generator

### Mic format
```
python audiovisual_synth.py mic
```

### EM32 format
```
python audiovisual_synth.py em32
```
