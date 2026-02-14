### Training

1. Organize videos into folders, one for each sign, for example:
videos/
  hello/
    video1.mp4
    video2.mp4
    ...
  yes/
    clip1.mp4
    ...
  thanks/
    recording1.mp4
    ...

2. Run the training script
python auto_train.py

Options:
--video_dir ./my_videos — point to a different folder
--update_signs — automatically updates the SIGNS list in main.py, api.py, training.py, and collect_data.py so everything stays in sync
--skip_extract — retrain from existing .npy files without re-processing videos
--epochs 200 — train longer if accuracy is low
--frames 30 — frames per sequence (matches your existing config)