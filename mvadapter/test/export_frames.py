from .utils.video import export_frames

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the extracted frames.")
    parser.add_argument("--frame_offset", type=int, default=0, help="Offset to apply to frame indices.")
    args = parser.parse_args()

    export_frames(args.video_path, args.output_dir, frame_offset=args.frame_offset)