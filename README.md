# SoundQ

### TODO

- [ ] All - to get access to CARP and start setting up our repo
- [ ] All - to start generating synthetic data and data augmentations under CARC
- [ ] Adrian - continue investigation on audiovisual saliency models and how we could integrate them to our model
- [ ] Rithik - video transformation sanity checks
  - [ ] Start looking into how we can integrate audio as well to match the video transformation 
- [ ] Bala - to study the `audiovisual_synth.py` script
  - [ ] Work on making moving sources following the same scripts and design pattern

### In Progress

- [ ] Bala working on how to work with the Co-DETR model to generate bounding boxes
- [ ] Rithik integrating the audiovisual data with the audio channel swapping augmentations 
- [ ] Adrian gathering other datasets to generate synthetic data. Currently we have the MUSIC dataset. Seeking to get VGGSound

### Done ✓

- [x] AudioVisual SELDnet model training. Adrian obtained baseline results
- [x] AudioVisual data synthesizer: generates video overlays and spatializes audio
- [x] Adrian - working on STAViS network integration with SELDNet: done but inference doesn't seem to be ideal. Requires lots of pre-processing and it is computationally expensive (takes ours to process the STARSS dataset). I have the extracted features in case we decide to use them.
