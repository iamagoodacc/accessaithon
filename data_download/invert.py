#!/usr/bin/env python3
"""
Video Horizontal Flip - Recursive Batch Processor (FFmpeg version)
Flips/mirrors all MP4 files in videos/ directory horizontally across the y-axis.
Uses FFmpeg for better compatibility with various video formats.
"""

import subprocess
import sys
from pathlib import Path


def check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_video_validity(input_path):
    """
    Check if a video file is valid using FFmpeg.
    
    Args:
        input_path: Path to the input video file
    
    Returns:
        bool: True if valid, False otherwise
    """
    cmd = [
        'ffmpeg',
        '-v', 'error',
        '-i', str(input_path),
        '-f', 'null',
        '-'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False
        )
        # If there are errors in stderr, the video might be problematic
        # But we'll be lenient - only fail if FFmpeg completely can't read it
        return result.returncode == 0
    except:
        return False


def flip_video_ffmpeg(input_path, output_path):
    """
    Flip a video file horizontally using FFmpeg.
    
    Args:
        input_path: Path to the input video file
        output_path: Path for the output file
    
    Returns:
        bool: True if successful, False otherwise
    """
    # FFmpeg command to flip video horizontally
    # hflip filter flips the video horizontally (mirror across y-axis)
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-vf', 'hflip',  # Horizontal flip filter
        '-c:a', 'copy',  # Copy audio without re-encoding
        '-y',  # Overwrite output file if it exists
        str(output_path)
    ]
    
    try:
        # Run FFmpeg with output suppression
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print(f"  ✓ Complete")
            return True
        else:
            print(f"  ✗ Error: FFmpeg failed")
            if result.stderr:
                # Print last few lines of error
                error_lines = result.stderr.strip().split('\n')
                for line in error_lines[-3:]:
                    if line.strip():
                        print(f"    {line}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def find_video_files(directory):
    """
    Recursively find all video files in a directory.
    
    Args:
        directory: Path to the directory to search
    
    Returns:
        list: List of Path objects for video files
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        return []
    
    # Find all common video file extensions
    video_extensions = ['*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mov', '*.MOV', 
                       '*.mkv', '*.MKV', '*.webm', '*.WEBM']
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(directory_path.rglob(ext))
    
    return video_files


def process_all_videos(videos_dir='videos'):
    """
    Process all video files in the videos directory recursively.
    Saves flipped videos in the same directory as the original with _invert suffix.
    
    Args:
        videos_dir: Path to the videos directory
    """
    # Check if FFmpeg is installed
    if not check_ffmpeg():
        print("Error: FFmpeg is not installed or not in PATH")
        print("\nInstall FFmpeg:")
        print("  macOS:   brew install ffmpeg")
        print("  Ubuntu:  sudo apt-get install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        return
    
    videos_path = Path(videos_dir)
    
    if not videos_path.exists():
        print(f"Error: Directory '{videos_dir}' not found.")
        print(f"Creating directory: {videos_dir}")
        videos_path.mkdir(parents=True, exist_ok=True)
        print(f"Please add video files to the '{videos_dir}' directory and run again.")
        return
    
    # Find all video files
    video_files = find_video_files(videos_dir)
    
    if not video_files:
        print(f"No video files found in '{videos_dir}' directory.")
        return
    
    print(f"Found {len(video_files)} video file(s) to process:\n")
    
    successful = 0
    failed = 0
    skipped = 0
    
    for i, input_file in enumerate(video_files, 1):
        # Get relative path from videos directory
        relative_path = input_file.relative_to(videos_path)
        
        # Create output path in the SAME directory as input
        # foo/bar/a.mp4 -> foo/bar/a_invert.mp4
        output_file = input_file.parent / f"{input_file.stem}_invert{input_file.suffix}"
        
        print(f"[{i}/{len(video_files)}] Processing: {relative_path}")
        
        # Show file size
        file_size = input_file.stat().st_size
        print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        # Check if output already exists
        if output_file.exists():
            print(f"  ⚠ Skipping: Output file already exists")
            skipped += 1
            print()
            continue
        
        # Quick validity check
        print(f"  Checking video validity...")
        if not check_video_validity(input_file):
            print(f"  ⚠ Skipping: Video appears to be corrupted or unreadable")
            skipped += 1
            print()
            continue
        
        # Process the video
        success = flip_video_ffmpeg(input_file, output_file)
        
        if success:
            successful += 1
            print(f"  Output: {output_file.relative_to(videos_path)}")
        else:
            failed += 1
        
        print()  # Empty line between files
    
    # Summary
    print("=" * 60)
    print(f"Processing complete!")
    print(f"  ✓ Successful: {successful}")
    if skipped > 0:
        print(f"  ⚠ Skipped: {skipped}")
    if failed > 0:
        print(f"  ✗ Failed: {failed}")
    print("=" * 60)


def main():
    """Main function to process all videos in videos/ directory."""
    print("=" * 60)
    print("Video Horizontal Flip - FFmpeg Batch Processor")
    print("=" * 60)
    print()
    
    videos_dir = 'videos'
    
    # Allow custom directory via command line
    if len(sys.argv) > 1:
        videos_dir = sys.argv[1]
    
    process_all_videos(videos_dir)


if __name__ == "__main__":
    main()
