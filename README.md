# SoundQ

## Features

- An audio-visual synthetic data generator with spatial audio and 360-degree video. 

- A suite of scripts to perform `data_augmentation` on 360-degree audio and video.

- An enhanced audio-visual SELDNet model with comparable performance to the [audio-only SELDNet23](https://github.com/sharathadavanne/seld-dcase2023)

  - The model integrates [Detic](https://github.com/facebookresearch/Detic), but any other detection model can also be integrated within the training pipeline.

## Installation

See [installation instructions](docs/INSTALL.md).

## Results on development dataset

We benchmark our model following the [DCASE Challenge 2023 Task3](https://dcase.community/challenge2023/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes) [SELD evaluation metric](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-localization-detection-and-tracking#h.ragsbsp7ujs).

The following table includes only the best performing system (as documented in [DCASE results](https://dcase.community/challenge2023/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes-results)). The evaluation metric scores for the test split of the development dataset is given below. 

| Model | Dataset | ER<sub>20°</sub> | F<sub>20°</sub> | LE<sub>CD</sub> | LR<sub>CD</sub> |
| ---- | ----| --- | --- | --- | --- |
| Audio-only SELDNet23 (baseline) | Ambisonic (FOA + Multi-ACCDOA) | 0.57 | 29.9 % | 21.6&deg; | 47.7 % |
| Audio-visual SELDNet23 (baseline) | Ambisonic + Video | 1.07 | 14.3  % | 48.0 &deg; | 35.5 % |
| ** Audio-visual SELDNet23 (ours) ** | Ambisonic (FOA + Multi-ACCDOA) + Video | 0.65 | 24.9 % | 18.7&deg; | 37.5 % |
