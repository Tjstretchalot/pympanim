"""Describes something which is capable of generating frames, potentially out
of order. Besides being able to generate frames out of order, everything is
single-processed from the perspective of a frame generator.

This module also contains various implementations of FrameGenerators that
are ultimately transformations of other FrameGenerators"""

import PIL.Image
import typing
import pytypeutils as tus
import pympanim.utils as mutils
from pympanim.easing import Easing

def img_to_bytes(img: PIL.Image) -> bytes:
    """Converts the given pillow image to a valid return value for
    FrameGenerator.generate_at.

    Args:
        img (PIL.Image): The image that was generated

    Returns:
        The rgba encoded bytes for the given pillow image
    """
    if img.mode == 'RGBA':
        return img.tobytes()
    return img.convert('RGBA').tobytes()

class FrameGenerator:
    """The interface for something that is capable of generating frames.
    This is completely context-free, in the sense that it should be able
    to generate frames for any valid time in any order with arbitrary
    delays or other frame generators being called. Prior to start this
    should be picklable, which may require implementing __getstate__ and
    __setstate__.
    """

    @property
    def duration(self) -> float:
        """Returns the duration of this generator. Describes the interval
        of time that frames can be generated for. generate_at will only
        be called for 0 <= time_ms <= duration. The duration is in milliseconds
        """
        raise NotImplementedError

    @property
    def frame_size(self) -> typing.Tuple[int, int]:
        """Gets the size of the frames that this generates in pixels.

        Returns:
            typing.Tuple[int, int]: The width and height of the generated
                frames in pixels.
        """
        raise NotImplementedError

    def start(self) -> None:
        """Called when this frame generator has been loaded onto the thread
        that will be used to generate frames.
        """
        pass

    def generate_at(self, time_ms: float) -> bytes:
        """Generates a frame that is time_ms milliseconds into the scene. Time
        in milliseconds is used instead of frame number because a frame
        generator ought to be agnostic to the frame rate.

        Note that img_to_bytes can be used to convert pillow images to the
        correct return value for this function.

        Arguments:
            time_ms (float): the elapsed time from the first frame of the
                frames that this generates.

        Returns:
            The image encoded in rgba format.
        """
        raise NotImplementedError

    def generate_at_pil(self, time_ms: float) -> PIL.Image:
        """Generates a frame that is time_ms milliseconds into the scene. By
        default this just wraps the result from generate_at. This is exposed
        in case a faster implementation is possible, such as when the frame
        generator actually operates on pillow images.

        Arguments:
            time_ms (float): the elapsed time from the first frame of the
                frames that this generates.

        Returns:
            The image in RGBA format.
        """
        return PIL.Image.frombytes(
            'RGBA', self.frame_size, self.generate_at(time_ms))

    def finish(self) -> None:
        """Called when this frame generator will not be used anymore."""
        pass

class SequenceFrameGenerator(FrameGenerator):
    """A frame generator that generates frames from other frame generators
    in a particular order. The first frame of a scene is used in preference
    to the last frame of a scene where there is ambiguity.

    Attributes:
        children (tuple[FrameGenerator]): The frame generators that do the
            heavy lifting, in the time-order.

        children_end_at (tuple[float]): For each child, this is the time
            relative to the sequence frame generator that the child generator
            ends. Used for performance
        _search_hint (int): This is "guessed" index in children for where
            frames will fall in. If this is correct, we can find the correct
            child in constant time. If it is wrong, we will fallback to a
            linear search. This makes use of the fact that while frame generators
            must work when frame times are uncorrelated, in practice they tend
            to come at least monotonically
    """
    def __init__(self, children: typing.Tuple[FrameGenerator]):
        tus.check(children=(children, (list, tuple)))
        tus.check_listlike(children=(children, FrameGenerator, (1, None)))

        self.children = tuple(children)
        self.children_end_at = []
        cur = 0
        frame_size = self.children[0].frame_size

        for i, child in enumerate(self.children):
            if child.frame_size != frame_size:
                raise ValueError(
                    f'children[0].frame_size = {frame_size}, but '
                    + f'children[{i}].frame_size = {child.frame_size}')
            cur += child.duration
            self.children_end_at.append(cur)

        self._search_hint = 0

    @property
    def duration(self):
        return self.children_end_at[-1]

    @property
    def frame_size(self):
        return self.children[0].frame_size

    def start(self):
        for child in self.children:
            child.start()

    def child_at(self, time_ms: float
                ) -> typing.Tuple[int, FrameGenerator, float]:
        """Returns the child and the time relative to the child that the given
        time relative to us corresponds to.

        Returns:
            (int): the index in children for the child that was found
            (FrameGenerator): the child that was found
            (float): the time relative to the child for the corresponding frame
        """
        i, time = mutils.find_child(self.children_end_at, time_ms,
                                    self._search_hint)
        return i, self.children[i], time


    def generate_at(self, time_ms: float) -> bytes:
        i, child, reltime = self.child_at(time_ms)
        self._search_hint = i
        return child.generate_at(reltime)

    def generate_at_pil(self, time_ms: float) -> PIL.Image:
        i, child, reltime = self.child_at(time_ms)
        self._search_hint = i
        return child.generate_at_pil(reltime)

    def finish(self):
        for child in self.children:
            child.finish()

class TimeRescaleFrameGenerator(FrameGenerator):
    """Takes another frame generator and runs it with a given playback rate

    Attributes:
        child (FrameGenerator): the thing which actually generates frames
        playback_rate (float): the number of milliseconds that pass for the
            child for every millisecond we see. 1.5 = 1.5x playback rate
    """
    def __init__(self, child: FrameGenerator, playback_rate: float):
        tus.check(
            playback_rate=(playback_rate, (int, float)),
            child=(child, FrameGenerator)
        )
        if playback_rate <= 0:
            raise ValueError(f'playback_rate={playback_rate} must be positive')
        self.playback_rate = float(playback_rate)
        self.child = child

    @property
    def duration(self):
        return self.child.duration / self.playback_rate

    @property
    def frame_size(self):
        return self.child.frame_size

    def start(self):
        self.child.start()

    def generate_at(self, time_ms: float):
        return self.child.generate_at(time_ms * self.playback_rate)

    def generate_at_pil(self, time_ms: float):
        return self.child.generate_at_pil(time_ms * self.playback_rate)

    def finish(self):
        self.child.finish()

class TimeRescaleExactDurationFrameGenerator(FrameGenerator):
    """Takes another frame generator and runs it with the given duration,
    linearly scaling time. Returns the exact duration it is passed in.

    Attributes:
        child (FrameGenerator): the thing which actually generates frames
        new_duration (float): the number of milliseconds we run the child for
    """
    def __init__(self, child: FrameGenerator, new_duration: float):
        tus.check(
            new_duration=(new_duration, (int, float)),
            child=(child, FrameGenerator)
        )
        if new_duration <= 0:
            raise ValueError(f'new_duration={new_duration} must be positive')
        self.new_duration = float(new_duration)
        self.child = child

    @property
    def duration(self):
        return self.new_duration

    @property
    def frame_size(self):
        return self.child.frame_size

    def start(self):
        self.child.start()

    def generate_at(self, time_ms: float):
        return self.child.generate_at(time_ms * (self.child.duration / self.new_duration))

    def generate_at_pil(self, time_ms: float):
        return self.child.generate_at_pil(time_ms * (self.child.duration / self.new_duration))

    def finish(self):
        self.child.finish()

class TimeDilateFrameGenerator(FrameGenerator):
    """Takes another frame generator and dilates time according to a specific
    easing.

    Attributes:
        dilator (Easing): the rule for transforming our percent progress to
            the childs percent progress
        dilator_kwargs (dict): the arguments to the dilator
        child (FrameGenerator): the thing which actually generates frames
    """
    def __init__(self, child: FrameGenerator, dilator: Easing,
                 dilator_kwargs: typing.Optional[dict] = None):
        tus.check(
            child=(child, FrameGenerator),
            dilator_kwargs=(dilator_kwargs, (dict, type(None)))
        )
        tus.check_callable(dilator=dilator)
        self.child = child
        self.dilator = dilator
        self.dilator_kwargs = (
            dilator_kwargs if dilator_kwargs is not None else dict())

    @property
    def duration(self):
        return self.child.duration

    @property
    def frame_size(self):
        return self.child.frame_size

    def start(self):
        self.child.start()

    def generate_at(self, time_ms: float):
        perc_time = time_ms / self.duration
        resc_perc_time = self.dilator(perc_time, **self.dilator_kwargs)
        resc_time = self.duration * resc_perc_time
        return self.child.generate_at(resc_time)

    def generate_at_pil(self, time_ms: float):
        perc_time = time_ms / self.duration
        resc_perc_time = self.dilator(perc_time, **self.dilator_kwargs)
        resc_time = self.duration * resc_perc_time
        return self.child.generate_at(resc_time)

    def finish(self):
        self.child.finish()

class TimeReverseFrameGenerator(FrameGenerator):
    """Takes another frame generator and plays it in reverse.

    Attributes:
        child (FrameGenerator): the frame generator to play in reverse
    """
    def __init__(self, child: FrameGenerator):
        tus.check(child=(child, FrameGenerator))
        self.child = child

    @property
    def duration(self):
        return self.child.duration

    @property
    def frame_size(self):
        return self.child.frame_size

    def start(self):
        self.child.start()

    def _child_time(self, time_ms: float) -> float:
        return self.duration - time_ms

    def generate_at(self, time_ms: float):
        return self.child.generate_at(self._child_time(time_ms))

    def generate_at_pil(self, time_ms: float):
        return self.child.generate_at_pil(self._child_time(time_ms))

    def finish(self):
        self.child.finish()


class RescaleFrameGenerator(FrameGenerator):
    """Takes another frame generator and rescales the image to the given
    size.

    Attributes:
        new_frame_size (tuple[int, int]): the new size of the frame
        resample (int): see PIL.Image.resize
        child (FrameGenerator): the frame generator which is being resized
    """
    def __init__(self, child: FrameGenerator,
                 new_frame_size: typing.Tuple[int, int],
                 resample: int = PIL.Image.LANCZOS):
        tus.check(
            child=(child, FrameGenerator),
            new_frame_size=(new_frame_size, (list, tuple)),
            resample=(resample, int)
        )
        tus.check_listlike(new_frame_size=(new_frame_size, int, 2))
        self.child = child
        self.new_frame_size = tuple(new_frame_size)
        self.resample = resample

    @property
    def duration(self):
        return self.child.duration

    @property
    def frame_size(self):
        return self.new_frame_size

    def start(self):
        self.child.start()

    def generate_at(self, time_ms: float):
        return img_to_bytes(self.generate_at_pil(time_ms))

    def generate_at_pil(self, time_ms):
        img = self.child.generate_at_pil(time_ms)
        return img.resize(self.new_frame_size, self.resample)

    def finish(self):
        self.child.finish()

class CroppedFrameGenerator(FrameGenerator):
    """Takes a frame generator and plays only a part of it.

    Attributes:
        crop_start (float): the time at which the child starts, relative to
            the child
        crop_end (float): the time at which the child ends, relative to the
            child.
        child (FrameGenerator): the frame generator which is being cropped
    """
    def __init__(self, child: FrameGenerator, crop_start: float,
                 crop_end: float) -> None:
        tus.check(
            child=(child, FrameGenerator),
            crop_start=(crop_start, float),
            crop_end=(crop_end, float)
        )
        if not 0 <= crop_start < child.duration:
            raise ValueError(
                f'crop_start={crop_start} not in [0, child.duration='
                + f'{child.duration})')
        if not crop_start < crop_end <= child.duration:
            raise ValueError(
                f'crop_end={crop_end} not in (crop_start={crop_start}'
                + f', child.duration={child.duration}]')

        self.child = child
        self.crop_start = crop_start
        self.crop_end = crop_end

    @property
    def duration(self):
        return self.crop_end - self.crop_start

    @property
    def frame_size(self):
        return self.child.frame_size

    def start(self):
        self.child.start()

    def _child_time(self, time_ms):
        return time_ms + self.crop_start

    def generate_at(self, time_ms: float):
        return self.child.generate_at(self._child_time(time_ms))

    def generate_at_pil(self, time_ms: float):
        return self.child.generate_at_pil(self._child_time(time_ms))

    def finish(self):
        self.child.finish()

class OverlayFrameGenerator(FrameGenerator):
    """Plays one frame generator over top another frame generator. This uses
    alpha compositing.

    Attributes:
        base (FrameGenerator): the frame generator that is underneath and
            determines the frame size
        overlay (FrameGenerator): the frame generator that is on top and
            applied over top the base
        pos (tuple[int, int]): the top left coordinate of the overlay relative
            to the top-left of the base
    """
    def __init__(self, base: FrameGenerator, overlay: FrameGenerator,
                 pos: typing.Tuple[int, int]):
        tus.check(base=(base, FrameGenerator), overlay=(overlay, FrameGenerator),
                  pos=(pos, (tuple, list)))
        tus.check_listlike(pos=(pos, int, 2))

        if (
                pos[0] < 0
                or pos[1] < 0
                or pos[0] + overlay.frame_size[0] > base.frame_size[0]
                or pos[1] + overlay.frame_size[1] > base.frame_size[1]):
            raise ValueError(f'cannot fit overlay at {pos} of size '
                             + f'{overlay.frame_size} onto frame of size '
                             + str(base.frame_size))

        if base.duration != overlay.duration:
            raise ValueError(f'durations dont match: base has {base.duration}'
                             + f' and overlay has {overlay.duration}')

        self.base = base
        self.overlay = overlay
        self.pos = pos

    @property
    def duration(self):
        return self.base.duration

    @property
    def frame_size(self):
        return self.base.frame_size

    def start(self):
        self.base.start()
        self.overlay.start()

    def generate_at(self, time_ms: float):
        return img_to_bytes(self.generate_at_pil(time_ms))

    def generate_at_pil(self, time_ms: float):
        bimg = self.base.generate_at_pil(time_ms)
        oimg = self.overlay.generate_at_pil(time_ms)
        bimg.alpha_composite(oimg, self.pos)
        return bimg

    def finish(self):
        self.base.finish()
        self.overlay.finish()


class FluentFG:
    """Describes a factory for frame generators that can stich them together
    with a fluent-style interface. Best understood through an example:

    ```
        import pympanim.frame_gen as fg
        import pympanim.easing as easing

        a: fg.FrameGenerator = # .. omitted ..
        b: fg.FrameGenerator = # .. omitted ..

        gen = fg.FluentFG(a)
            .crop(15, 30, 's')
            .dilate(easing.smoothstep)
            .then(
                fg.FluentFG(b)
                    .crop(1, 2, 'm')
                    .time_rescale(2)
                    .dilate(easing.squeeze, {amt: 0.3})
                    .build()
            )
            .then(
                fg.FluentFG(a).crop(0, 15, 's').build()
            )
            .build()
    ```

    If you create your own class that accepts a child followed by additional
    arguments, then you can still use it in almost a fluent style with the
    apply function.

    Attributes:
        base (FrameGenerator): the base frame generator that we are building,
            ignoring the sequence of frame generators (to avoid excess wrapping
            when in sequences)
        followed_by (list[FrameGenerator]): the frame generators that should
            come after this one in sequence.
    """
    # Less wrapping could be done if we take into account multiple crops can be
    # combined into a single crop, eg
    # fg.FluentFG(a).crop(15, 30, 's').crop(0, 5, 's') is the same as
    # fg.FluentFG(a).crop(15, 20, 's') but it is not clear that this is a very
    # common situation. This holds for the other transforms as well

    def __init__(self, base: FrameGenerator) -> None:
        self.base = base
        self.followed_by = []

    def then(self, other: FrameGenerator) -> 'FluentFG':
        """Ensures that other will be the next frame in the sequence after
        base.
        """
        tus.check(other=(other, FrameGenerator))
        if other.frame_size != self.base.frame_size:
            raise ValueError(
                f'cannot then({other}) when other.frame_size='
                + f'{other.frame_size} and the current frame size is '
                + f'{self.base.frame_size}')

        self.followed_by.append(other)
        return self

    def apply(self, transform: typing.Callable,
              *args, **kwargs) -> 'FluentFG':
        """Applies the given transformation to the currently build frame
        generator, using the given arguments

        Args:
            transform (typing.Callable): Must accept FrameGenerator, *args,
                **kwargs and return a FrameGenerator
        """
        tus.check_callable(transform=transform)
        if self.followed_by:
            self.base = self.build()
            self.followed_by = []

        self.base = transform(self.base, *args, **kwargs)
        tus.check(transform_result=(self.base, FrameGenerator))
        return self

    def duration(self):
        """Returns the duration of the current video"""
        result = self.base.duration
        for other in self.followed_by:
            result += other.duration
        return result

    def frame_size(self):
        """Returns the frame size of the current video"""
        return self.base.frame_size

    def reverse(self) -> 'FluentFG':
        """Takes the current video and reverses time"""
        return self.apply(TimeReverseFrameGenerator)

    def crop(self, start: float, end: float, unit: str) -> 'FluentFG':
        """Crops this to be in the given subsection of time, where time
        is given in the specific unit.

        Args:
            start (float): the start time of the currently described video
            end (float): the end time of the currently described video
            unit (str): the unit of time, i.e. 'ms', 's', 'h', 'd'. For
                all possible keys, see list(pympanim.utils.UNITS)
        """
        tus.check(
            start=(start, (int, float)),
            end=(end, (int, float)),
            unit=(unit, str)
        )
        if unit not in mutils.UNITS_LOOKUP:
            raise ValueError(
                f'unit={unit} must be one of {list(mutils.UNITS_LOOKUP)}')

        ms_per_unit = mutils.UNITS_LOOKUP[unit]
        start *= ms_per_unit
        end *= ms_per_unit

        if start < 0:
            raise ValueError(f'start={start} must be positive')
        if start >= end:
            raise ValueError(f'start={start} must be before end={end}')
        if end >= self.base.duration:
            raise ValueError(
                f'end={end} is after current duration={self.base.duration}')

        return self.apply(CroppedFrameGenerator, start, end)

    def dilate(self, dilator: Easing,
               dilator_kwargs: typing.Optional[dict] = None) -> 'FluentFG':
        """Rescales time for the current frame generator to be rescaled by
        the given dilator.

        Args:
            dilator (Easing): the rule for changing time
            dilator_kwargs (typing.Optional[dict]): if specified, the keyword
                arguments to the dilator
        """
        tus.check(
            dilator_kwargs=(dilator_kwargs, (dict, type(None)))
        )
        tus.check_callable(dilator=dilator)

        return self.apply(TimeDilateFrameGenerator, dilator, dilator_kwargs)

    def rescale(self, new_width: int, new_height: int,
                resample: int = PIL.Image.LANCZOS) -> 'FluentFG':
        """Rescales the frame generator to the given width and height, using
        the specified resampling technique.

        Args:
            new_width (int): the new width of the output
            new_height (int): the new height of the output
            resample (int, optional): Resampling technique.
                Defaults to PIL.Image.LANCZOS.
        """
        tus.check(
            new_width=(new_width, int),
            new_height=(new_height, int),
            resample=(resample, int)
        )

        return self.apply(RescaleFrameGenerator, (new_width, new_height), resample)

    def time_rescale(self, playback_rate: float) -> 'FluentFG':
        """Rescales time to play back at the given rate. For example,
        a playback_rate of 2 means that the final video will complete in
        50% of the time.
        """
        tus.check(playback_rate=(playback_rate, (int, float)))

        return self.apply(TimeRescaleFrameGenerator, playback_rate)

    def time_rescale_exact(self, new_duration: float, unit: str) -> 'FluentScene':
        """Similar to time_rescale except instead of specifying an exact
        playback rate, which can cause rounding issues on the new
        duration, you instead specify an exact new duration and accept some
        rounding on the playback rate.

        Args:
            new_duration (float): the new duration in the given unit
            unit (str): one of 'ms', 's', 'min', 'hr'
        """
        tus.check(
            new_duration=(new_duration, (int, float)),
            unit=(unit, str)
        )
        if unit not in mutils.UNITS_LOOKUP:
            raise ValueError(f'unknown unit \'{unit}\'; should be one of '
                             + str(list(mutils.UNITS_LOOKUP)))

        ms_per_unit = mutils.UNITS_LOOKUP[unit]
        return self.apply(TimeRescaleExactDurationFrameGenerator,
                          new_duration*ms_per_unit)

    def overlay(self, overlay: FrameGenerator, pos: typing.Tuple[int, int]):
        """Overlays the current frame generator with the given one at the given
        position. Note that the overlayed frame generator must have the same
        duration as the current one and must fit entirely within the frame
        """
        tus.check(overlay=(overlay, FrameGenerator),
                  pos=(pos, (list, tuple)))
        tus.check_listlike(pos=(pos, int, 2))

        return self.apply(OverlayFrameGenerator, overlay, pos)

    def build(self) -> FrameGenerator:
        """Returns the frame generator that was described using a fluent api
        """
        if self.followed_by:
            scenes = [self.base]
            scenes.extend(self.followed_by)
            return SequenceFrameGenerator(scenes)
        return self.base
