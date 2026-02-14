#!/usr/bin/env python3
"""
Download ASL video clips from a JSON index and organize them into
folders named after each gloss (sign/gesture).

Requirements:
    pip install yt-dlp requests

Usage:
    python download_asl_videos.py --input dataset.json --output ./videos

    # Split work across 3 workers:
    python download_asl_videos.py -i dataset.json -o ./videos --worker 1/3
    python download_asl_videos.py -i dataset.json -o ./videos --worker 2/3
    python download_asl_videos.py -i dataset.json -o ./videos --worker 3/3

    # Or manually specify a range of gloss entries:
    python download_asl_videos.py -i dataset.json -o ./videos --start 0 --end 500
    python download_asl_videos.py -i dataset.json -o ./videos --start 500 --end 1000
"""

import argparse
import json
import os
import csv
import subprocess
import shutil
import requests
from pathlib import Path
from urllib.parse import urlparse


def is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def is_swf_url(url: str) -> bool:
    return url.lower().endswith(".swf")


def sanitize_filename(name: str) -> str:
    """Remove or replace characters that are problematic in file/folder names."""
    return "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name).strip()


def download_youtube(url: str, output_path: str, frame_start: int, frame_end: int, fps: int) -> bool:
    """Download a YouTube video (or a segment of it) using yt-dlp."""
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4/best",
        "--merge-output-format", "mp4",
        "-o", output_path,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]

    # If there are specific frame boundaries, download only that segment
    if frame_start > 1 or frame_end > 0:
        start_sec = max(0, (frame_start - 1)) / fps
        cmd += ["--download-sections", f"*{start_sec:.2f}-"]
        if frame_end > 0:
            end_sec = frame_end / fps
            # Use external downloader section syntax
            cmd = [
                "yt-dlp",
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4/best",
                "--merge-output-format", "mp4",
                "-o", output_path,
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                "--download-sections", f"*{start_sec:.2f}-{end_sec:.2f}",
            ]

    try:
        subprocess.run(cmd, check=True, timeout=120)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"    yt-dlp error: {e}")
        return False


def download_direct(url: str, output_path: str) -> bool:
    """Download a video file directly via HTTP."""
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    Download error: {e}")
        return False


def trim_video(input_path: str, output_path: str, frame_start: int, frame_end: int, fps: int) -> bool:
    """Trim a directly-downloaded video to the relevant frame range using ffmpeg."""
    start_sec = max(0, (frame_start - 1)) / fps
    cmd = ["ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-i", input_path]
    if frame_end > 0:
        end_sec = frame_end / fps
        duration = end_sec - start_sec
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-c", "copy", output_path]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Download ASL videos from JSON index")
    parser.add_argument("--input", "-i", required=True, help="Path to the JSON index file")
    parser.add_argument("--output", "-o", default="./videos", help="Output root directory (default: ./videos)")
    parser.add_argument("--skip-swf", action="store_true", default=True, help="Skip .swf Flash files (default: True)")
    parser.add_argument("--trim", action="store_true", default=False,
                        help="Trim non-YouTube videos to frame_start/frame_end using ffmpeg")
    parser.add_argument("--start", type=int, default=None,
                        help="Start index (inclusive) into the gloss list, 0-based")
    parser.add_argument("--end", type=int, default=None,
                        help="End index (exclusive) into the gloss list, 0-based")
    parser.add_argument("--worker", type=str, default=None,
                        help="Worker slice in N/M format (e.g. 1/3 = first of 3 workers). "
                             "Automatically divides the dataset into M equal chunks.")
    args = parser.parse_args()

    # Load the JSON index
    with open(args.input, "r") as f:
        dataset = json.load(f)

    # --- Determine which slice of the dataset this worker handles ---
    total_glosses = len(dataset)

    if args.worker:
        if args.start is not None or args.end is not None:
            parser.error("Cannot use --worker together with --start/--end")
        try:
            worker_num, num_workers = map(int, args.worker.split("/"))
        except ValueError:
            parser.error("--worker must be in N/M format, e.g. 1/3")
        if worker_num < 1 or worker_num > num_workers:
            parser.error(f"Worker number must be between 1 and {num_workers}")
        chunk_size = total_glosses // num_workers
        remainder = total_glosses % num_workers
        # Distribute remainder across first `remainder` workers
        start = sum(chunk_size + (1 if i < remainder else 0) for i in range(worker_num - 1))
        end = start + chunk_size + (1 if (worker_num - 1) < remainder else 0)
    else:
        start = args.start if args.start is not None else 0
        end = args.end if args.end is not None else total_glosses

    start = max(0, start)
    end = min(total_glosses, end)
    dataset = dataset[start:end]

    print(f"Processing glosses [{start}:{end}] of {total_glosses} total ({len(dataset)} entries)")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    # Check for required tools
    has_ytdlp = shutil.which("yt-dlp") is not None
    has_ffmpeg = shutil.which("ffmpeg") is not None

    if not has_ytdlp:
        print("WARNING: yt-dlp not found. YouTube videos will be skipped.")
        print("  Install with: pip install yt-dlp")
    if not has_ffmpeg and args.trim:
        print("WARNING: ffmpeg not found. Trimming will be disabled.")
        args.trim = False

    total = 0
    downloaded = 0
    skipped = 0
    failed = 0

    for entry in dataset:
        gloss = entry["gloss"]
        folder_name = sanitize_filename(gloss)
        gloss_dir = output_root / folder_name
        gloss_dir.mkdir(parents=True, exist_ok=True)

        # Write/append metadata for this gloss
        metadata_path = gloss_dir / "metadata.csv"
        write_header = not metadata_path.exists()

        with open(metadata_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow([
                    "instance_id", "video_id", "source", "signer_id",
                    "variation_id", "split", "fps",
                    "frame_start", "frame_end",
                    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                    "url", "filename", "status"
                ])

            for inst in entry.get("instances", []):
                total += 1
                url = inst["url"]
                instance_id = inst["instance_id"]
                source = inst.get("source", "unknown")
                video_id = inst.get("video_id", "unknown")
                fps = inst.get("fps", 25)
                frame_start = inst.get("frame_start", 1)
                frame_end = inst.get("frame_end", -1)
                bbox = inst.get("bbox", [0, 0, 0, 0])

                filename = f"{gloss}_{instance_id:04d}_{source}_{video_id}.mp4"
                filepath = gloss_dir / filename

                # Skip if already downloaded
                if filepath.exists():
                    print(f"  [EXISTS] {filename}")
                    writer.writerow([
                        instance_id, video_id, source, inst.get("signer_id"),
                        inst.get("variation_id"), inst.get("split"), fps,
                        frame_start, frame_end, *bbox, url, filename, "exists"
                    ])
                    downloaded += 1
                    continue

                # Skip SWF files
                if is_swf_url(url) and args.skip_swf:
                    print(f"  [SKIP]   {filename} (Flash .swf)")
                    writer.writerow([
                        instance_id, video_id, source, inst.get("signer_id"),
                        inst.get("variation_id"), inst.get("split"), fps,
                        frame_start, frame_end, *bbox, url, filename, "skipped_swf"
                    ])
                    skipped += 1
                    continue

                # Download
                print(f"  [{total}] Downloading {filename} ...")
                success = False

                if is_youtube_url(url):
                    if has_ytdlp:
                        success = download_youtube(url, str(filepath), frame_start, frame_end, fps)
                    else:
                        print(f"    Skipping YouTube URL (yt-dlp not installed)")
                        skipped += 1
                        writer.writerow([
                            instance_id, video_id, source, inst.get("signer_id"),
                            inst.get("variation_id"), inst.get("split"), fps,
                            frame_start, frame_end, *bbox, url, filename, "skipped_no_ytdlp"
                        ])
                        continue
                else:
                    success = download_direct(url, str(filepath))

                    # Optionally trim to frame range
                    if success and args.trim and (frame_start > 1 or frame_end > 0):
                        trimmed_path = gloss_dir / f"trimmed_{filename}"
                        if trim_video(str(filepath), str(trimmed_path), frame_start, frame_end, fps):
                            os.replace(str(trimmed_path), str(filepath))
                            print(f"    Trimmed to frames {frame_start}-{frame_end}")

                status = "ok" if success else "failed"
                if success:
                    downloaded += 1
                else:
                    failed += 1

                writer.writerow([
                    instance_id, video_id, source, inst.get("signer_id"),
                    inst.get("variation_id"), inst.get("split"), fps,
                    frame_start, frame_end, *bbox, url, filename, status
                ])

    worker_label = f" (worker {args.worker})" if args.worker else f" (glosses [{start}:{end}])"
    print(f"\nDone{worker_label}! Total: {total} | Downloaded: {downloaded} | Skipped: {skipped} | Failed: {failed}")
    print(f"Videos saved to: {output_root.resolve()}")


if __name__ == "__main__":
    main()
