"""A video of a red square crossing the screen"""

import time
import typing
import PIL.Image
import pympanim.frame_gen as fg
import pympanim.worker as pmaw
import pympanim.easing as easing
import os
import argparse


class RedSquareCrosses(fg.FrameGenerator):
    """Moves a red square from the left to the right side in unit time"""

    def __init__(self, frame_size: typing.Tuple[int, int], slowdown_sleep: float):
        self._frame_size = frame_size
        self._slowdown_sleep = slowdown_sleep

    @property
    def duration(self):
        return 1

    @property
    def frame_size(self):
        return self._frame_size

    def generate_at(self, time_ms):
        with self.generate_at_pil(time_ms) as img:
            return fg.img_to_bytes(img)

    def generate_at_pil(self, time_ms):
        if self._slowdown_sleep is not None:
            time.sleep(self._slowdown_sleep)
        box_size = int(self._frame_size[1] * 0.2)

        box_x = int(time_ms * (self._frame_size[0] - box_size))
        box_y = int((self._frame_size[1] / 2) - (box_size / 2))

        img = PIL.Image.new("RGBA", self._frame_size, "black")
        with PIL.Image.new("RGBA", (box_size, box_size), "red") as box:
            img.paste(box, (box_x, box_y))
        return img


def _main():
    parser = argparse.ArgumentParser(
        description="Generate a video of a red square crossing the screen"
    )
    parser.add_argument(
        "--out", type=str, default="out/examples/redsquare.mp4", help="Output file path"
    )
    parser.add_argument(
        "--mkdirs",
        action="store_true",
        help="Create the output directory if it does not exist",
    )
    parser.add_argument(
        "--replace", action="store_true", help="Replace the output file if it exists"
    )
    parser.add_argument(
        "--duration", type=int, default=5, help="Duration of the video in seconds"
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Width of the video in pixels"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Height of the video in pixels"
    )
    parser.add_argument(
        "--fps", type=int, default=60, help="Frames per second of the video"
    )
    parser.add_argument(
        "--dpi", type=int, default=100, help="Dots per inch of the video"
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=-1,
        help="Bitrate of the video in kilobits per second. 0 or negative for unconstrained",
    )
    parser.add_argument(
        "--initial-workers",
        type=int,
        help="Number of initial worker processes, unset for auto based on CPU count",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of worker processes, unset for auto based on CPU count",
    )
    parser.add_argument(
        "--initial-frames-per-sync",
        type=int,
        help="Initial frames before forcing workers to sync to check delay. Will be auto-tuned, unset for default",
    )
    parser.add_argument(
        "--min-frames-per-sync",
        type=int,
        help="Minimum frames before forcing workers to sync to check delay, unset for default",
    )
    parser.add_argument(
        "--max-frames-per-sync",
        type=int,
        help="Maximum frames before forcing workers to sync to check delay, unset for default based on CPU count",
    )
    parser.add_argument(
        "--slowdown-sleep",
        type=float,
        help="Sleep for this many seconds per frame to slow down frame generation (default 0)",
        default=0.0,
    )
    args = parser.parse_args()

    if args.mkdirs:
        out_folder = os.path.dirname(args.out)
        os.makedirs(out_folder, exist_ok=True)

    if os.path.exists(args.out):
        if args.replace:
            os.remove(args.out)
        else:
            raise FileExistsError(
                f"Output file {args.out} already exists (use --replace to replace)"
            )

    print(f"Starting on PID {os.getpid()}")
    pmaw.produce(
        fg.FluentFG(RedSquareCrosses((args.width, args.height), args.slowdown_sleep))
        .time_rescale_exact(args.duration, "s")
        .dilate(easing.smootheststep)
        .build(),
        args.fps,
        args.dpi,
        args.bitrate,
        args.out,
        settings=pmaw.PerformanceSettings(
            num_workers=args.initial_workers,
            max_workers=args.max_workers,
            **(
                {"frames_per_sync": args.initial_frames_per_sync}
                if args.initial_frames_per_sync is not None
                else {}
            ),
            **(
                {"min_frames_per_sync": args.min_frames_per_sync}
                if args.min_frames_per_sync is not None
                else {}
            ),
            **(
                {"max_frames_per_sync": args.max_frames_per_sync}
                if args.max_frames_per_sync is not None
                else {}
            ),
        ),
    )


if __name__ == "__main__":
    _main()
