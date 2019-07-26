"""This module stiches together encoded images using ffmpeg into videos.
"""
import pytypeutils as tus
import typing
import os
import time
from collections import deque

import subprocess as sp

class ImageSticherPerformanceHandler:
    """An object which is interested in the performance of an ImageSticher.
    This can be used to determine the current balance between image generating
    and image encoding.
    """

    def post_work(self, frames_rec: int, frames_proc: int, nooo: int):
        """Called after the image sticher performs some work.

        Args:
            frames_rec (int): The number of frames this received from workers
            frames_proc (int): The number of frames processed to the ffmpeg
            nooo (int): The number of frames waiting to be processed
        """
        raise NotImplementedError

class ISRunningAveragePerfHandler(ImageSticherPerformanceHandler):
    """This maintains a running average over time for number of frames received
    and number of frames processed. If the number of frames received exceeds
    the number of frames processed significantly, it may be wise to reduce
    the number of workers on image generation. Alternatively, if we have
    processed as many or more images than we have generated, then depending
    on if there is a backlog or not it could be necessary to increase the
    number of workers generating images.

    Attributes:
        window_size (float): the number of seconds wide this window is
        frames_rec (deque[(int, float)]): Every tuple is (number, timestamp)
            for when the image sticher received some number of images
        sum_frames_rec (int): the number of frames received in the current
            window
        frames_proc (deque[(int, float)]): Every tuple is (number, timestamp)
            for when the image sticher processed some numebr of images
        sum_frames_proc (int): the number of frames processed in the current
            window

        _first_enc_at (float): the earliest time we saw a frame encoded at. If
            this wasn't at least window_size away, then we dont have a full
            window of information available.
    """
    def __init__(self, window_size: float):
        tus.check(window_size=(window_size, (int, float)))

        self.window_size = window_size
        self.frames_rec = deque()
        self.sum_frames_rec = 0
        self.frames_proc = deque()
        self.sum_frames_proc = 0

        self._first_enc_at = None

    def clean_window(self) -> None:
        """Goes through and removes all values in frames_rec and frames_proc
        that are outside of the window"""
        prune_before = time.time() - self.window_size
        while self.frames_rec:
            left = self.frames_rec.popleft()
            if left[1] >= prune_before:
                self.frames_rec.appendleft(left)
                break
            self.sum_frames_rec -= left[0]

        while self.frames_proc:
            left = self.frames_proc.popleft()
            if left[1] >= prune_before:
                self.frames_proc.appendleft(left)
                break
            self.sum_frames_proc -= left[0]

    def have_window(self) -> bool:
        """Returns True if we have a full window of information available,
        False otherwise"""
        return (
            self._first_enc_at is not None
            and (time.time() > self._first_enc_at + self.window_size)
        )

    def mean(self) -> typing.Tuple[float, float]:
        """Returns the number of frames received per second and the number of
        frames processed per second in the current window.
        """
        self.clean_window()
        return (
            (self.sum_frames_rec / self.window_size),
            (self.sum_frames_proc / self.window_size)
        )

    def post_work(self, frames_rec: int, frames_proc: int, nooo: int):
        self.clean_window()
        if frames_rec > 0:
            self.frames_rec.append((frames_rec, time.time()))
            self.sum_frames_rec += frames_rec
        if frames_proc > 0:
            self.frames_proc.append((frames_proc, time.time()))
            self.sum_frames_proc += frames_proc


class ImageSticher:
    """Describes a multiprocessing-friendly ffmpeg binding. The idea behind
    this is to allow processing frames that are coming from potentially many
    other processes. Frames are sent to this via potentially many queues, which
    are processed somewhat evenly.

    Frames which arrive out of order are stored in memory until they are ready
    to be used. Having a large number of out of order frames could mean that
    we are generating images too fast, we are generating images out of order,
    or do_work is not being called often enough.

    Attributes:
        frame_size (tuple[int, int]): the size of each frame in pixels
        dpi (int or float): the number of pixels per inch of the figure
        bitrate (int): maximum output bitrate. May be less than or equal
            to 0 for unconstrained bitrate. In kilobytes.
        fps (int): number of frames in a single second
        outfile (str): where we are saving to. May be specified without an
            extension to default to mp4.

        ffmpeg_proc (mp.Process): the process we use to communicate with ffmpeg

        next_frame (int): the index for the next frame to write to file. If you
            know how many frames are in the movie, this can be used to know
            when all frames have been processed. Its value will be
            the number of frames in the movie when done, since it is 0-indexed.
        ooo_frames (dict[int, bytes]): frames that have arrived but have not
            been sent to the ffmpeg process yet.

        receive_queues (list[queue-like]): the queue-like objects that we are
            receiving frames from.

        perfs (list[ImageSticherPerformanceHandler]): things interested in
            the performance of this object. May be modified externally.

        max_ooo_frames (int): the maximum number of out of order frames before
            we raise an error
        block_size (int): the number of bytes we send to the ffmpeg process at
            a time, at most. Greatly impacts performance.
    """

    def __init__(self, frame_size: typing.Tuple[int, int],
                 dpi: typing.Union[int, float], bitrate: int,
                 fps: int, outfile: str,
                 max_ooo_frames: int = 5000,
                 block_size: int = 4048) -> None:
        tus.check(
            frame_size=(frame_size, (list, tuple)),
            dpi=(dpi, (int, float)), bitrate=(bitrate, int),
            fps=(fps, int), outfile=(outfile, str),
            max_ooo_frames=(max_ooo_frames, (int, float)),
            block_size=(block_size, int)
        )
        tus.check_listlike(frame_size=(frame_size, (int, float), 2))

        wo_ext, ext = os.path.splitext(outfile)
        if ext == '':
            outfile = wo_ext + '.mp4'
        elif ext != '.mp4':
            raise NotImplementedError(f'only mp4 encoding is supported, but got {ext}')

        if os.path.exists(outfile):
            raise FileExistsError(outfile)

        self.frame_size = frame_size
        self.dpi = dpi
        self.bitrate = bitrate
        self.fps = fps
        self.outfile = outfile
        self.ffmpeg_proc = None
        self.next_frame = 0
        self.ooo_frames = dict()
        self.max_ooo_frames = max_ooo_frames
        self.block_size = block_size
        self.receive_queues = []
        self.perfs = []

    def _spawn_ffmpeg(self) -> None:
        """Spawns the ffmpeg process"""
        if self.ffmpeg_proc is not None:
            raise RuntimeError('_spawn_ffmpeg called when ffmpeg_proc is '
                               + f'{self.ffmpeg_proc} (not None)')

        args = ['ffmpeg', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{self.frame_size[0]}x{self.frame_size[1]}',
                '-pix_fmt', 'rgba', '-r', str(self.fps),
                '-loglevel', 'quiet',
                '-i', 'pipe:0',
                '-vcodec', 'h264', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart']

        if self.bitrate > 0:
            args.extend(['-b', f'{self.bitrate}k'])
        args.extend(['-y', self.outfile])

        create_flags = sp.CREATE_NO_WINDOW if 'nt' in os.name else 0
        self.ffmpeg_proc = sp.Popen(args, shell=False, stdout=None, stderr=None,
                                    stdin=sp.PIPE, creationflags=create_flags)

    def _cleanup_ffmpeg(self) -> None:
        """Cleans up the ffmpeg process. This will wait for it to terminate"""
        self.ffmpeg_proc.communicate()
        self.ffmpeg_proc = None

    def start(self) -> None:
        """Prepares this image sticher to begin. Receive queues may still be
        added or removed at any time."""
        self._spawn_ffmpeg()

    def register_queue(self, queue) -> None:
        """Registers the specified queue-like object as something frames can
        be received from. Must have a get_nowait and empty member."""
        if queue is None:
            raise ValueError('queue is None')
        if not hasattr(queue, 'empty'):
            raise ValueError(f'queue {queue} is missing empty member')
        if not hasattr(queue, 'get_nowait'):
            raise ValueError(f'queue {queue} is missing get_nowait member')
        self.receive_queues.append(queue)

    def remove_queue(self, queue) -> None:
        """Removes the given queue-like object from this. Uses the same
        comparison as list.remove. Raises ValueError if the queue is not
        currently being used to fetch frames."""
        self.receive_queues.remove(queue)

    def check_queues(self) -> int:
        """Checks for items from each of the receive queues and pushes them
        onto the local memory dict. Returns the number of frames received"""

        nframes = 0

        for queue in self.receive_queues:
            if not queue.empty():
                nframes += 1
                frame, img_bytes = queue.get_nowait()

                if frame < self.next_frame:
                    raise ValueError('received frame we already processed! '
                                     + f'got {frame}, at {self.next_frame}')
                if frame in self.ooo_frames:
                    raise ValueError(f'received duplicate frame: {frame}')

                self.ooo_frames[frame] = img_bytes
                if len(self.ooo_frames) > self.max_ooo_frames:
                    raise ValueError('exceeded maximum frame cache (now have '
                                     + f'{len(self.ooo_frames)} frames waiting)')

        return nframes

    def process_frame(self) -> bool:
        """Processes the next frame to the ffmpeg process if it is available.
        Returns True if we processed a frame, False if we did not."""
        if self.next_frame not in self.ooo_frames:
            return False

        img_bytes = self.ooo_frames.pop(self.next_frame)

        for kb_start in range(0, len(img_bytes), self.block_size):
            self.ffmpeg_proc.stdin.write(
                img_bytes[kb_start:kb_start + self.block_size])

        self.next_frame += 1
        return True

    def do_work(self):
        """A catch-all function to do some work. Returns True if some work
        was done, False otherwise"""
        recv = self.check_queues()
        proc = 1 if self.process_frame() else 0
        for perf in self.perfs:
            perf.post_work(recv, proc, len(self.ooo_frames))
        return recv > 0 or proc > 0

    def finish(self):
        """Cleanly closes handles"""
        if self.ffmpeg_proc is not None:
            self._cleanup_ffmpeg()
