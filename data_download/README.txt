### Usage
python download_asl_videos.py --input your_dataset.json --output ./videos

### Split Work

# Person 1:
python download_asl_videos.py -i dataset.json -o ./videos --worker 1/3

# Person 2:
python download_asl_videos.py -i dataset.json -o ./videos --worker 2/3

# Person 3:
python download_asl_videos.py -i dataset.json -o ./videos --worker 3/3

### Requirements
pip install yt-dlp requests