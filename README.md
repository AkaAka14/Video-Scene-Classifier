# Video Scene Classifier
A Deep Learning based system to automatically classify video segments into four categories: **News, Sports, Movies, and Violence.**

## Architecture
This project uses a hybrid approach:
*   **Feature Extraction:** GoogLeNet (Inception v1) pre-trained on ImageNet to extract spatial features from video frames.
*   **Temporal Logic:** A Long Short-Term Memory (LSTM) network to process the sequence of frames and understand the "action" over time.

## Project Structure
- `data_loader.py`: Handles video frame extraction and preprocessing.
- `model.py`: Contains the GoogLeNet + LSTM architecture.
- `train.py`: Script for model training.
- `test_video.py`: Script to run inference on new video files.

## Current Status
- **Pipeline:** Functional.
- **Initial Training:** Completed with 10 videos per class.
- **Current Challenge:** Improving accuracy (Requires more diverse dataset).
