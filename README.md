# Multimodal Hierarchical Bayesian Retention Model

With the widespread availability of video-sharing platforms and smart devices, there is a wealth of multimodal video data on the internet, containing textual, auditory, and visual information conveying diverse emotional cues. Analyzing emotional states and identifying emotions from this data is of significant societal importance. However, single-modal data present challenges in this regard. Hence, this research focuses on leveraging multimodal data for emotion analysis and recognition.

## Overview

This repository contains the code for a Multimodal Hierarchical Bayesian Retention Model based on a hierarchical Bayesian chain network framework. The model effectively integrates multimodal features, reduces feature redundancy, and enhances inter-modality correlations and intrinsic modality features through retention modules and self-attention mechanisms. To amplify differences among various emotions and sentiments, contrastive learning is employed, enhancing the model’s analysis capabilities.

## Code Branches

### `HBFN_nodata`
- `ensembling.py`: Testing code.
- `infonce.py`: Contrastive learning loss function.
- `mosei_datset.py`: Data reading from MOSEI dataset.
- `new_config.py`: Parameter configuration.
- `new_main.py`: Code execution entry point.
- `new_model_8.py`: Model 1 implementation.
- `new_model_12.py`: Model 2 implementation.
- `new_train.py`: Training code.
- `requirements.txt`: Required libraries for installation.

### Directories
- `data/MOSEI`: Contains `train_align.pkl`, `test_align.pkl`, and `valid_align.pkl` datasets.
- `layers`: Implementation of different layers (`fc.py`, `layer_norm.py`).
- `utils`: Utility scripts (`compute_args.py`, `plot.py`, `pred_func.py`, `tokenize.py`).

## Usage

To use the provided code:
1. Set up the MOSEI dataset.
2. Install the required libraries listed in `requirements.txt`.
3. Execute `python new_main.py` to start training.
4. Run `python ensembling.py` to test model accuracy.

## Performance

The model trained and evaluated on the MOSEI dataset outperforms the TDN model:
- 1.2% increase in accuracy for binary sentiment analysis.
- 0.3% improvement for seven-class sentiment analysis.
- An average uplift of over 2% in emotion recognition.

Feel free to explore and use this codebase for your multimodal emotion analysis tasks.
