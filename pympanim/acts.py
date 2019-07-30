"""An act is a particular type of frame generator that has some state which it
can generate images from. An act has one or more scenes, which manipulate the
state that the act renders rather than the images themselves.

For example, a matplotlib animation is naturally understood as an act under
this formulation. Consider a scatter plot animated through time. The act would
set up the axes and initialize the scatter plot, and the scenes would just
move where the scatter points are.
"""

import pytypeutils as tus
import pympanim.frame_gen as fgen
import pympanim.utils as mutils
from pympanim.easing import Easing
import typing
import PIL.Image

class ActState:
    """The interface for things that can act as state in acts, which is shared
    amongst the scenes."""
    pass

class ActRenderer:
    """Something which is capable of rendering a particular act state subtype
    """
    @property
    def frame_size(self) -> typing.Tuple[int, int]:
        """Returns the size in pixels that this renders the state at"""
        raise NotImplementedError

    def render(self, act_state: ActState) -> bytes:
        """Renders the given act state to raw rgba bytes"""
        raise NotImplementedError

    def render_pil(self, act_state: ActState) -> PIL.Image:
        """Renders the given act state to a pillow image. If there are nested
        renderers which all use pillow images it will typically faster to use
        this chain then constantly converting to/from bytes. By default, this
        just wraps the result from render.
        """
        return PIL.Image.frombytes(
            'RGBA', self.frame_size, self.render(act_state))

class Scene:
    """Describes a scene in an act, which manipulates the state of the act rather
    than the image directly.
    """
    @property
    def duration(self) -> float:
        """Returns the number of milliseconds that this scene lasts for"""
        return 1

    def start(self, act_state: ActState):
        """Called when this scene is loaded onto the thread that will
        ultimately use the scene to render frames, but before apply has
        been called"""
        pass

    def enter(self, act_state: ActState):
        """Called when this scene is about to be rendered, and the last scene
        in the act was not this scene.
        """
        pass

    def apply(self, act_state: ActState, time_ms: float, dbg: bool = False):
        """Must update the state such that it will render the frame at the
        given time relative to this scene."""
        pass

    def exit(self, act_state: ActState):
        """Called when this was the last scene in the act to be rendered, but
        the act is about to render a different scene."""
        pass

# wow, you're a real
class Act(fgen.FrameGenerator):
    """Describes an act, which describes some state through an ActState that
    can be displayed. An act is composed of scenes which modify the ActState
    rather than images.

    Attributes:
        state (ActState): the underlying state for this act
        renderer (ActRenderer): the thing which can render the state
        scenes (tuple[Scene]): the scenes that this make up this act

        scenes_end_at (tuple[int]): when the scenes end at. for performance
        _scene_hint (int): where we will begin linear seaerches at when
            going from a scene time to a scene.
        _last_scene (int, optional): the last scene that we rendered.
    """
    def __init__(self, state: ActState, renderer: ActRenderer,
                 scenes: typing.Tuple[Scene]):
        tus.check(
            state=(state, ActState),
            renderer=(renderer, ActRenderer),
            scenes=(scenes, (list, tuple))
        )
        tus.check_listlike(scenes=(scenes, Scene))
        self.state = state
        self.renderer = renderer
        self.scenes = tuple(scenes)

        scenes_end_at = []
        cur = 0
        for scene in scenes:
            cur += scene.duration
            scenes_end_at.append(cur)
        self.scenes_end_at = scenes_end_at
        self._scene_hint = 0
        self._last_scene = None

    @property
    def duration(self):
        return self.scenes_end_at[-1]

    @property
    def frame_size(self):
        return self.renderer.frame_size

    def scene_at(self, time_ms: float) -> typing.Tuple[int, Scene, float]:
        """Returns the scene that occurs at the given time in the act and the
        relative time within that scene that the given time corresponds with.

        Arguments:
            time_ms (float): the time within the act that you want the scene of

        Returns:
            (int): the index within scenes for the scene
            (Scene): the actual scene it corresponds with
            (float): the time relative to the scene that the given time
                corresponds to
        """
        i, rtime = mutils.find_child(self.scenes_end_at, time_ms,
                                     self._scene_hint)
        return i, self.scenes[i], rtime

    def setup_state(self, time_ms: float) -> None:
        """Sets the current act state to represent the state at the
        given time.

        Args:
            time_ms (float): The time that you want the state to be in
        """
        ind, scene, reltime = self.scene_at(time_ms)

        if ind != self._last_scene:
            if self._last_scene:
                self.scenes[self._last_scene].exit(self.state)
            scene.enter(self.state)
            self._last_scene = ind

        scene.apply(self.state, reltime)

    def start(self):
        for scene in self.scenes:
            scene.start(self.state)

    def generate_at(self, time_ms: float) -> bytes:
        self.setup_state(time_ms)
        return self.renderer.render(self.state)

    def generate_at_pil(self, time_ms: float) -> PIL.Image:
        self.setup_state(time_ms)
        return self.renderer.render_pil(self.state)

    def finish(self):
        if self._last_scene:
            self.scenes[self._last_scene].exit(self.state)
            self._last_scene = None

class SceneSequenceScene(Scene):
    """Describes a single scene which is made up of several other scenes.
    It may be more convenient to tie scenes together in this manner within a
    single act rather than have multiple acts, if they all share the same
    ActState and ActRenderer.

    Attributes:
        scenes (tuple[Scene]): the scenes that make up this scene
        scenes_end_at (tuple[int]): when the scenes that make up this scene end
            at, calculated once for performance reasons
        _scene_hint (int): where we start linear searches at within scenes
        _last_scene (int, optional): the last active scene, None if this
            sequence scene was not the last scene to be rendered.
    """
    def __init__(self, scenes: typing.Tuple[Scene]):
        tus.check(scenes=(scenes, (list, tuple)))
        tus.check_listlike(scenes=(scenes, Scene))
        self.scenes = tuple(scenes)

        scenes_end_at = []
        cur = 0
        for scene in scenes:
            cur += scene.duration
            scenes_end_at.append(cur)
        self.scenes_end_at = tuple(scenes_end_at)
        self._scene_hint = 0
        self._last_scene = None

    @property
    def duration(self):
        return self.scenes_end_at[-1]

    def start(self, act_state: ActState):
        for scene in self.scenes:
            scene.start(act_state)

    def apply(self, act_state: ActState, time_ms: float, dbg: bool = False):
        ind, reltime = mutils.find_child(self.scenes_end_at, time_ms,
                                         self._scene_hint)
        self._scene_hint = ind
        scene = self.scenes[ind]
        if dbg:
            print(f'sequence at time {time_ms} applying child {ind} '
                  + f'at time {reltime}')

        if ind != self._last_scene:
            if self._last_scene:
                self.scenes[self._last_scene].exit(act_state)
            scene.enter(act_state)
            self._last_scene = ind

        scene.apply(act_state, reltime, dbg)


    def exit(self, act_state: ActState):
        if self._last_scene:
            self.scenes[self._last_scene].exit(act_state)
            self._last_scene = None

class JoinedScene(Scene):
    """Describes multiple scenes which are applied at the same time in a
    particular order. Helpful when you have scenes that do different things.

    Attributes:
        children (tuple[Scene]): the children of this scene in a particular order
        sep_enters (bool): if true, the children each get a proper enter/exit
            as per the documentation. if false, the children are all entered
            and exitted at the same time which may improve performance,
            especially when you know that the children being joined do not
            interfere with each other.
    """
    def __init__(self, children: typing.Tuple[Scene],
                 sep_enters: bool) -> None:
        tus.check(children=(children, (list, tuple)),
                  sep_enters=(sep_enters, bool))
        tus.check_listlike(children=(children, Scene, (1, None)))

        duration = children[0].duration

        for i, child in enumerate(children):
            if child.duration != duration:
                raise ValueError(
                    f'children[0].duration={duration}, but '
                    + f'children[{i}].duration={child.duration}'
                )

        self.children = tuple(children)
        self.sep_enters = sep_enters

    @property
    def duration(self):
        return self.children[0].duration

    def start(self, act_state: ActState):
        for child in self.children:
            child.start(act_state)

    def enter(self, act_state: ActState):
        if not self.sep_enters:
            for child in self.children:
                child.enter(act_state)

    def apply(self, act_state: ActState, time_ms: float, dbg: bool = False):
        for i, child in enumerate(self.children):
            if self.sep_enters:
                child.enter(act_state)
            if dbg:
                print(f'JoinedScene applying children[{i}]={type(child)} '
                      + f'at time {time_ms}')
            child.apply(act_state, time_ms, dbg)
            if self.sep_enters:
                child.exit(act_state)

    def exit(self, act_state: ActState):
        if not self.sep_enters:
            for child in self.children:
                child.exit(act_state)

class TimeRescaleScene(Scene):
    """Describes a scene which is just another scene played back at a different
    rate.

    Attributes:
        playback_rate (float): the number of seconds of time relative to the
            child that corresponds to 1 second relative to us.
        child (Scene): the actual scene that is used
    """
    def __init__(self, child: Scene, playback_rate: float) -> None:
        tus.check(
            child=(child, Scene),
            playback_rate=(playback_rate, (int, float))
        )
        if playback_rate <= 0:
            raise ValueError(
                f'playback_rate={playback_rate} should be positive')

        self.child = child
        self.playback_rate = playback_rate

    @property
    def duration(self):
        return self.child.duration / self.playback_rate

    def start(self, act_state: ActState):
        self.child.start(act_state)

    def enter(self, act_state: ActState):
        self.child.enter(act_state)

    def apply(self, act_state: ActState, time_ms: float, dbg: bool = False):
        newtime = time_ms * self.playback_rate
        if dbg:
            print(f'time rescale at {time_ms} applying child at {newtime}')
        self.child.apply(act_state, newtime, dbg)

    def exit(self, act_state: ActState):
        self.child.exit(act_state)

class TimeRescaleExactDurationScene(Scene):
    """Acts very similarly to a time rescale scene except this has an exact
    duration that it uses, moving the floating point error to the playback
    rate instead.

    Attributes:
        new_duration (float): the duration that we have
        child (Scene): the child scene whose duration is changed
    """
    def __init__(self, child: Scene, new_duration: float):
        tus.check(child=(child, Scene),
                  new_duration=(new_duration, (int, float)))
        self.new_duration = new_duration
        self.child = child

    @property
    def duration(self):
        return self.new_duration

    def start(self, act_state: ActState):
        self.child.start(act_state)

    def enter(self, act_state: ActState):
        self.child.enter(act_state)

    def apply(self, act_state: ActState, time_ms: float, dbg: bool = False):
        newtime = time_ms * (self.child.duration / self.new_duration)
        if dbg:
            print(f'time dur. resc. at {time_ms} applying child at {newtime}')
        self.child.apply(act_state, newtime, dbg)

    def exit(self, act_state: ActState):
        self.child.exit(act_state)

class TimeReverseScene(Scene):
    """Reverses time for the given child scene.

    Attributes:
        child (Scene): the actual child scene
    """
    def __init__(self, child: Scene) -> None:
        tus.check(child=(child, Scene))
        self.child = child

    @property
    def duration(self):
        return self.child.duration

    def start(self, act_state: ActState):
        self.child.start(act_state)

    def enter(self, act_state: ActState):
        self.child.enter(act_state)

    def apply(self, act_state: ActState, time_ms: float, dbg: bool = False):
        newtime = self.duration - time_ms
        if dbg:
            print(f'time reverse at {time_ms} applying child at {newtime}')
        self.child.apply(act_state, newtime, dbg)

    def exit(self, act_state: ActState):
        self.child.exit(act_state)

class TimeDilateScene(Scene):
    """Describes a scene which is another scene played for the same duration,
    but time moves at a different rate each point according to a specific
    easing.

    Attributes:
        dilator (Easing): the rule for mapping % time in relative to this scene
            to % time relative to the child scene
        dilator_kwargs (dict): keyword arguments for the dilator
        child (Scene): the scene which is actually running
    """
    def __init__(self, child: Scene, dilator: Easing, dilator_kwargs: dict = None):
        tus.check(child=(child, Scene),
                  dilator_kwargs=(dilator_kwargs, (dict, type(None))))
        tus.check_callable(dilator=dilator)
        self.dilator = dilator
        self.dilator_kwargs = (
            dict() if dilator_kwargs is None else dilator_kwargs)
        self.child = child

    @property
    def duration(self):
        return self.child.duration

    def start(self, act_state: ActState):
        self.child.start(act_state)

    def enter(self, act_state: ActState):
        self.child.enter(act_state)

    def apply(self, act_state: ActState, time_ms: float, dbg: bool = False):
        perc_time = time_ms / self.duration
        resc_perc_time = self.dilator(perc_time, **self.dilator_kwargs)
        resc_time = self.duration * resc_perc_time
        if dbg:
            print(f'time dilate at {time_ms} (perc: {perc_time}) applying '
                  + f'child at {resc_time} (perc: {resc_perc_time})')
        self.child.apply(act_state, resc_time, dbg)

    def exit(self, act_state: ActState):
        self.child.exit(act_state)

class CroppedScene(Scene):
    """Describes a scene which comes from taking only a segment of the child
    scene.

    Attributes:
        crop_start (float): the time relative to the child that the cropped
            scene starts at
        crop_end (float): the time relative to the child that the cropped scene
            ends at.
        child (Scene): the scene which is cropped
    """
    def __init__(self, child: Scene, crop_start: float, crop_end: float):
        tus.check(
            child=(child, Scene),
            crop_start=(crop_start, (int, float)),
            crop_end=(crop_end, (int, float))
        )
        if crop_start < 0:
            raise ValueError(f'crop_start={crop_start} should be positive')
        if crop_end <= crop_start:
            raise ValueError(
                f'crop_end - crop_start = {crop_end} - {crop_start} = '
                + f'{crop_end - crop_start} should be positive'
            )
        if crop_end > child.duration:
            raise ValueError(
                f'crop_end = {crop_end} should not exceed child.duration='
                + f'{child.duration}'
            )
        self.child = child
        self.crop_start = crop_start
        self.crop_end = crop_end

    @property
    def duration(self):
        return self.crop_end - self.crop_start

    def start(self, act_state: ActState):
        self.child.start(act_state)

    def enter(self, act_state: ActState):
        self.child.enter(act_state)

    def apply(self, act_state: ActState, time_ms: float, dbg: bool = False):
        newtime = time_ms + self.crop_start
        if dbg:
            print(f'crop at {time_ms} applying child at {newtime}')
        self.child.apply(act_state, newtime, dbg)

    def exit(self, act_state: ActState):
        self.child.exit(act_state)

class FluentScene:
    """Acts as a factory for scenes that are built in a fluent manner. For any
    scene transforms which are not included here, apply can be used for much
    the same effect.

    Attributes:
        base (Scene): the main scene that the transforms effect
        followed_by (list[Scene]): the scenes that should have been appended to
            base but haven't yet. This is to avoid excessive wrapping from
            sequences of scenes.
        joined_with (list[Scene]): the scenes that should have been joined to
            the base but haven't yet. Similar idea to followed_by
        join_is_sep (bool): True if the join enters occur separately, false
            otherwise.
        pushed (list[FluentScene]): scenes which are currently not being modified
            by commands.
    """
    def __init__(self, base: Scene):
        tus.check(base=(base, Scene))
        self.base = base
        self.followed_by = []
        self.joined_with = []
        self.join_is_sep = False
        self.pushed = []

    def duration(self):
        """Gets the current duration of this scene"""
        result = self.base.duration
        for scene in self.followed_by:
            result += scene.duration
        return result

    def apply(self, transform: typing.Callable, *args,
              **kwargs) -> 'FluentScene':
        """Applies the given transform to the current working scene. The
        transform should accept a Scene as its first argument, and then we
        pass it the specified arguments and keyword-arguments.

        Args:
            transform (typing.Callable): The transformation to apply
        """
        self.base = transform(self.build(), *args, **kwargs)
        self.followed_by = []
        self.joined_with = []
        return self

    def then(self, scene: Scene) -> 'FluentScene':
        """Has the given scene follow the currently described scene"""
        tus.check(scene=(scene, Scene))
        if self.joined_with:
            self.apply(lambda x: x)
        self.followed_by.append(scene)
        return self

    def push(self, scene: Scene) -> 'FluentScene':
        """Puts the currently generated scene into the background for now,
        meaning it will not be modified by future commands, and then begins
        the current state with the given scene.

        Example:
            scene1: Scene
            scene2: Scene
            acts.FluentScene(scene1)
                .time_rescale_exact(5, 's') # modifies scene1
                .push(scene2)
                .time_rescale_exact(5, 's') # only modifies scene2
                .pop() # now its scene 1 (5s) -> scene2 (5s)
                .dilate(easing.smoothstep) # modifies both scene 1 and 2
        """
        curscene = FluentScene(self.base)
        curscene.followed_by = self.followed_by
        curscene.joined_with = self.joined_with
        curscene.join_is_sep = self.join_is_sep

        self.followed_by = []
        self.joined_with = []
        self.join_is_sep = []
        self.pushed.append(curscene)
        self.base = scene
        return self

    def pop(self, style: str = 'then') -> 'FluentScene':
        """The analogue to push. Causes future calls to also modify the
        last pushed scene.
        """
        tus.check(style=(style, str))

        popped = self.pushed.pop()
        cur = self.build()

        self.base = popped.base
        self.followed_by = popped.followed_by
        self.joined_with = popped.joined_with
        self.join_is_sep = popped.join_is_sep
        if style == 'then':
            return self.then(cur)
        if style == 'join':
            return self.join(cur, False)
        if style == 'join_sep':
            return self.join(cur, True)
        raise ValueError(f'style={style} unsupported, should be then, join, or join_sep')

    def join(self, scene: Scene, sep_enters: bool) -> 'FluentScene':
        """Has the given scene apply at the same time as the current one.

        If sep_enters is true, the code path will be
            current.enter()
            current.apply()
            current.exit()
            scene.enter()
            scene.apply()
            scene.exit()

        Otherwise,
            current.enter()
            scene.enter()

            current.apply() # this
            scene.apply()   # and this may repeat multiple times

            current.exit()
            scene.exit()
        """
        if self.followed_by or (self.joined_with and
                                sep_enters != self.join_is_sep):
            self.apply(lambda x: x)
        self.joined_with.append(scene)
        self.join_is_sep = sep_enters
        return self

    def reverse(self) -> 'FluentScene':
        """Causes the current segment to be played in reverse."""
        return self.apply(TimeReverseScene)

    def crop(self, start: float, end: float, unit: str) -> 'FluentScene':
        """Crops this scene to start and end in the specified times relative
        to the current scene, using the given unit. The unit should be 'ms',
        's', or any other key in pympanim.utils.UNITS_LOOKUP
        """
        tus.check(start=(start, (int, float)),
                  end=(end, (int, float)),
                  unit=(unit, str))

        if unit not in mutils.UNITS_LOOKUP:
            raise ValueError(f'unknown unit \'{unit}\'; should be one of '
                             + str(list(mutils.UNITS_LOOKUP)))

        ms_per_unit = mutils.UNITS_LOOKUP[unit]

        real_start = start * ms_per_unit
        real_end = end * ms_per_unit
        return self.apply(CroppedScene, real_start, real_end)

    def dilate(self, dilator: Easing,
               dilator_kwargs: typing.Optional[dict] = None) -> 'FluentScene':
        """Transforms time according to the given easing.

        Args:
            dilator (Easing): The rule for manipulating time
            dilator_kwargs (dict, optional): Arguments to the dilator.
        """
        tus.check(
            dilator_kwargs=(dilator_kwargs, (dict, type(None))))
        tus.check_callable(dilator=dilator)

        return self.apply(TimeDilateScene, dilator, dilator_kwargs)

    def time_rescale(self, playback_rate: float) -> 'FluentScene':
        """Changes the playback rate, e.g. 2 for the result to have half the
        duration.

        Args:
            playback_rate (float): number of current seconds that should
                correspond to a single second.
        """
        tus.check(playback_rate=(playback_rate, (int, float)))
        return self.apply(TimeRescaleScene, playback_rate)

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
        return self.apply(TimeRescaleExactDurationScene, new_duration*ms_per_unit)

    def build(self):
        """Builds the actual scene that has been created by this factory"""
        res = self.base
        if self.followed_by:
            scenes = [res]
            scenes.extend(self.followed_by)
            res = SceneSequenceScene(scenes)
        if self.joined_with:
            scenes = [res]
            scenes.extend(self.joined_with)
            res = JoinedScene(scenes, self.join_is_sep)
        return res
