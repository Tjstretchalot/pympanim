"""A video of a red square crossing the screen"""

import typing
import PIL.Image
import pympanim.frame_gen as fg
import pympanim.worker as pmaw
import pympanim.easing as easing
import os

class RedSquareCrosses(fg.FrameGenerator):
    def __init__(self, frame_size: typing.Tuple[int, int]):
        self._frame_size = frame_size

    @property
    def duration(self):
        return 1

    @property
    def frame_size(self):
        return self._frame_size

    def generate_at(self, time_ms):
        return fg.img_to_bytes(self.generate_at_pil(time_ms))

    def generate_at_pil(self, time_ms):
        box_size = int(self._frame_size[1] * 0.2)

        box_x = int(time_ms * (self._frame_size[0] - box_size))
        box_y = int((self._frame_size[1] / 2) - (box_size / 2))

        img = PIL.Image.new('RGBA', self._frame_size, 'black')
        box = PIL.Image.new('RGBA', (box_size, box_size), 'red')
        img.paste(box, (box_x, box_y))
        return img

def _main():
    os.makedirs('out/examples', exist_ok=True)

    pmaw.produce(
        fg.FluentFG(RedSquareCrosses((640, 480)))
        .time_rescale(0.0002) # 5 seconds
        .dilate(easing.smootheststep)
        .build(),
        60,
        100,
        -1,
        'out/examples/redsquare.mp4'
    )

if __name__ == '__main__':
    _main()
