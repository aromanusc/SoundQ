# SoundQ — Enhanced sound event localization and detection in real 360-degree audio-visual soundscapes.

## Features

- An audio-visual synthetic data generator with spatial audio and 360-degree video. 

- A suite of scripts to perform `data_augmentation` on 360-degree audio and video.

  - Integrating audio channel swapping (ACS) as per [Wang et al.](https://arxiv.org/abs/2101.02919)

  - Integrating video pixel swapping (VPS) as per [Wang et al.](https://dcase.community/documents/challenge2023/technical_reports/DCASE2023_Du_102_t3.pdf)

- An enhanced audio-visual SELDNet model with comparable performance to the [audio-only SELDNet23](https://github.com/sharathadavanne/seld-dcase2023)

  - The model integrates [Detic](https://github.com/facebookresearch/Detic), but any other detection model can also be integrated within the training pipeline.

## Installation

See [installation instructions](docs/INSTALL.md).

## Results on development dataset

We benchmark our model following the [DCASE Challenge 2023 Task3](https://dcase.community/challenge2023/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes) [SELD evaluation metric](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-localization-detection-and-tracking#h.ragsbsp7ujs).

The following table includes only the best performing system (as documented in [DCASE results](https://dcase.community/challenge2023/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes-results)). The evaluation metric scores for the test split of the development dataset is given below. 

| Model | Dataset | ER<sub>20°</sub> | F<sub>20°</sub> | LE<sub>CD</sub> | LR<sub>CD</sub> |
| ---- | ----| --- | --- | --- | --- |
| AO SELDNet23 (baseline) | Ambisonic\* | 0.57 | 29.9 % | 21.6&deg; | 47.7 % |
| AV SELDNet23 (baseline) | Ambisonic + Video | 1.07 | 14.3  % | 48.0 &deg; | 35.5 % |
| **AV SELDNet23 (ours)** | Ambisonic\* + Video | 0.65 | 24.9 % | 18.7&deg; | 37.5 % |

Legend: AO=audio-only, AV=audio-visual, FOA=first order ambisonics format, \*=FOA + Multi-ACCDOA

## Citation

If you find our work useful, please cite our paper:
```
@article{roman2024enhanced,
  title={Enhanced Sound Event Localization and Detection in Real 360-degree audio-visual soundscapes},
  author={Roman, Adrian S and Balamurugan, Baladithya and Pothuganti, Rithik},
  journal={arXiv preprint arXiv:2401.17129},
  year={2024}
}
```
