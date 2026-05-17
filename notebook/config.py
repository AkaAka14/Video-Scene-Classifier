import os

DATASET_PATH = os.path.join("data", "frames")
VIDEO_PATH = os.path.join("data", "videos")
MODEL_PATH = os.path.join("weights", "scene_model.pth")

CLASSES = ["sports", "news", "movies", "violence"]

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
FRAME_SKIP = 10

DEVICE = "cuda"