# Multiprocessed Animations

## Summary

This library helps building movies in python from images. Specifically, it
intends to allow multiple threads to produce images which are stitched together
into movies with ffmpeg.

## Use-cases

This can be used to multithread the image creation process when using
matplotlib animations. FFMpeg will use multithreading to encode the images into
a video, however producing the images themselves is a bottleneck if you have
many cores available. Hence, this will vastly reduce the time it takes to
generate matplotlib animations.

This can also be used for generating scenes with
[PIL](https://pillow.readthedocs.io/en/stable/).

This library's goal is to make generating videos as simple as possible first,
then to go as fast as possible within those simple techniques. This gives fast
enough performance for many projects, and has the enormous benefit that you
can throw more hardware at the problem. Using ffmpeg directly without
multithreading image generation will not scale to more hardware.

## Performance

With ideal settings, the images should be generated at a rate that just barely
does not fill the ffmpeg process input pipe. This will ensure that images are
being generated as quickly as they can be encoded.

By default, this library will attempt to find the settings that accomplish this
task. This takes a bit of time to accomplish, so the final settings are exposed
and it can be helpful to use those when re-running roughly the same task. The
correct settings will depend on how long it takes to generate images and how
long it takes to encode them which varies based on the image statistics.

## Installation

`pip install pympanim`

## Dependencies

This depends on ffmpeg being installed. It can be installed
[here](https://ffmpeg.org/download.html). Other python dependencies will be
automatically installed by pip.

## Usage - Acts

### Motivation

Many times you have something which is capable of rendering some underlying
state in a consistent way, and the video you want to produce is made up of
parts that manipulate the state that is sent to the renderer. This use case
is handled specifically by `pympanim/acts.py`.

### Summary

Define the state that completely describes how to render things in a class that
subclasses `pympanim.acts.ActState`. Then create the thing which can render
the given state to an image (described via rgba bytes or a Pillow Image).

With these ready, create (one or more) `Scene`s, which are things that
manipulate the state you just created. A scene has some duration, and must be
able to set the state to correspond to a particular time within the scene.

This library will provide common manipulations of scenes - nonlinear time
manipulations, cropping, reversing, and sequencing. To make the most use of
these manipulations, Scenes should be as simple as possible.

Multiple Acts can be combined in much the same way. See Usage - Frame
Generators for details below.

To produce the video, use `pympanim.worker.produce`, creating a frame generator
out of the scenes using `pympanim.acts.Act`

### Boilerplate

```py
import PIL.Image
import pympanim.worker as pmaw
import pympanim.acts as acts
import pympanim.frame_gen as fg
import os

class MyActState(acts.ActState):
    pass

class MyActRenderer(acts.ActRenderer):
    @property
    def frame_size(self):
        return (1920, 1080) # in pixels

    def render(self, act_state: MyActState) -> bytes:
        return fg.img_to_bytes(self.render_pil(act_state))

    def render_pil(self, act_state: MyActState) -> PIL.Image:
        # By default, render_pil delegates to render. It is shown reversed
        # here for completeness.
        return PIL.Image.new('RGBA', self.frame_size, 'white')

class Scene1(acts.Scene):
    @property
    def duration(self):
        return 1 # 1 millisecond; allows you to treat time_ms as % progress

    def apply(self, act_state: MyActState, time_ms: float, dbg: bool = False):
        # dbg is passed across the scene heirarchy and will be false when
        # rendering the video. you may find it helpful to print some debug
        # information when it is true
        pass

def _scene():
    scene = Scene1()
    return (acts.FluentScene(scene)
            .dilate(lambda x: x**2) # any easing works here. see
                                    # pympanim.easing and pytweening
            .time_rescale(1 / 10000) # 10s of real time corresponds to
                                     # 1ms of scene time
            .build()
           )

def _main():
    os.makedirs('out', exist_ok=True)
    act_state = MyActState()
    act_renderer = MyActRenderer()

    pmaw.produce(
        acts.Act(act_state, renderer, [_scene()]),
        fps=60,
        dpi=100,
        bitrate=-1,
        outfile='out/video.mp4'
    )

if __name__ == '__main__':
    _main()

```

For method-level documentation, use the built-in `help` command, i.e.,
```
>python3
>>> import pympanim.acts
>>> help(pympanim.acts)
Help on module pympanim.acts in pympanim:
.. (omitted for readme brevity) ..
```

## Usage - Frame Generators

### Motivation

The most novel part of this library is the boilerplate to generate videos where
the image generation itself is multithreaded. This is exposed in as raw a manner
as possible using `pympanim/frame_gen.py` which is merely wrapped by
`pympanim/acts.py`. This section discusses how to use this library with minimal
abstraction.

### Summary

Create a subclass of `pympanim.frame_gen.FrameGenerator`, which requires
defining a duration, frame size in pixels, and a `generate_at` function which
can generate an image given just the time within the video.

This library provides common manipulations of vidoes that you can wrap your
video with, such as cropping, time dilations, reversing time, and combinations
of frame generators.

To produce the video, use `pympanim.worker.produce`. Everything else is handled
for you, including reasonable guesses for performance settings and runtime
performance tuning.

### Boilerplate

```py
import PIL.Image
import pympanim.frame_gen as fg
import pympanim.worker as pmaw
import os

class MyFrameGenerator(fg.FrameGenerator):
    @property
    def duration(self):
        return 1 # 1 ms, allows you to treat time_ms as % progress

    @property
    def frame_size(self):
        return (1920, 1080) # in pixels

    def generate_at(self, time_ms):
        # by default generate_at_pil delegates to generate_at. we show the
        # reverse for completeness
        return fg.img_to_bytes(self.generate_at_pil(time_ms))

    def generate_at_pil(self, time_ms):
        # this from white to red, you can do whatever
        return PIL.Image.new('RGBA', self.frame_size, f'#{int(time_ms*255):02x}0000')

def _fg():
    base = MyFrameGenerator()

    # random example of the stuff you can do
    return (fg.FluentFG(base)
            .time_rescale(1 / 10000)         # 10s long
            .then(                           # after current
                fg.FluentFG(base)            # play again
                    .time_rescale(1 / 10000) # also 10s long
                    .reverse()               # but this time in reverse
                    .build()
            )
            .build())

def _main():
    os.makedirs('out', exist_ok=True)

    pmaw.produce(
        _fg(),
        fps=60,
        dps=100,
        bitrate=-1,
        outfile='out/video.mp4'
    )

if __name__ == '__main__':
    _main()

```


## Examples

The examples/ folder has the sourcecode for the following examples:

```
python3 -m examples.redsquare
```

Produces https://gfycat.com/digitalmaleflyingfish

```
python3 -m examples.mpl_line
```

Produces https://gfycat.com/wickedpracticalattwatersprairiechicken

## Known Issues

If processes are forked instead of spawned, text will appear spastic in
matplotlib. In general, matplotlib does not handle forked processes well.
The worker attempts to force spawning mode instead of fork-mode, but this
will fail if the process is already forked. If you are experiencing weird
or glitchy videos, include this *before any other imports* which might be
spawning processes (e.g. torch or torchvision).

```py
import multiprocessing as mp

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass
```
