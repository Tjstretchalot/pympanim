"""Animates a matplotlib plot by increasing the frequency os a simple
cosine plot.
"""

import os
import pympanim.worker as pmaw
import pympanim.acts as acts
import pytweening
import pytypeutils as tus
import matplotlib.pyplot as plt
import numpy as np
import typing
import io

class CosData(acts.ActState):
    """Describes the state of the cosine data

    Attributes:
        frequency (float): the frequency for the plot
        indeps (ndarray[float]): the independent values to show
    """
    def __init__(self, frequency: float, indeps: np.ndarray):
        tus.check(frequency=(frequency, (int, float)))
        tus.check_ndarrays(
            indeps=(indeps, ('n_samples',), ('float32', 'float64'))
        )
        self.frequency = frequency
        self.indeps = indeps

class CosRenderer(acts.ActRenderer):
    """Capable of rendering a CosData scene

    Attributes:
        frame_size_inches (tuple[float, float]): the size of the frame in
            inches
        dpi (int): number of pixels per inch

        _frame_size (tuple[int, int]): the size of the frame in pixels
    """
    def __init__(self, frame_size_inches: typing.Tuple[float, float],
                 dpi: int):
        tus.check(frame_size_inches=(frame_size_inches, (list, tuple)),
                  dpi=(dpi, int))
        tus.check_listlike(frame_size_inches=(frame_size_inches, float, 2))
        self.frame_size_inches = tuple(frame_size_inches)
        self.dpi = dpi

        self._frame_size = (int(frame_size_inches[0] * dpi),
                            int(frame_size_inches[1] * dpi))

        # correct for rounding
        frame_size_inches = (self._frame_size[0] / dpi,
                             self._frame_size[1] / dpi)

    @property
    def frame_size(self):
        return self._frame_size

    def render(self, act_state: CosData):
        deps = np.cos(act_state.indeps * act_state.frequency)

        fig, ax = plt.subplots()
        ax.set_title(f'Frequency={act_state.frequency:.3f}')
        ax.plot(act_state.indeps, deps, linewidth=2, fmt='-r')

        hndl = io.BytesIO()
        fig.set_size_inches(*self.frame_size_inches)
        fig.savefig(hndl, format='rgba', dpi=self.dpi)
        res = hndl.getvalue()

        fig.close()
        return res

class SweepFrequencyScene(acts.Scene):
    """Sweeps frequency from 1 to 10 over 1ms"""
    @property
    def duration(self):
        return 1

    def apply(self, act_state: CosData, time_ms: float):
        act_state.frequency = 1 + time_ms*9

def _main():
    os.makedirs('out/examples', exist_ok=True)
    act_state = CosData(1, np.linspace(-1, 1, 100))
    renderer = CosRenderer((19.2, 10.8), 100)
    scene = SweepFrequencyScene()

    pmaw.produce(
        acts.Act(
            act_state, renderer,
            [
                acts.FluentScene(scene)
                .time_rescale(1 / 5000)
                .then(
                    acts.FluentScene(scene)
                    .reverse()
                    .dilate(pytweening.easeOutCubic)
                    .time_rescale(1 / 3000)
                    .build()
                )
                .then(
                    acts.FluentScene(scene)
                    .dilate(pytweening.easeInOutCirc)
                    .crop(0, 0.1, 'ms')
                    .time_rescale(1 / 1000)
                    .build()
                )
                .then(
                    acts.FluentScene(scene)
                    .reverse()
                    .dilate(pytweening.easeInOutCirc)
                    .crop(0, 0.1, 'ms')
                    .reverse()
                    .time_rescale(1 / 1000)
                    .build()
                )
                .build()
            ]
        ),
        60,
        100,
        -1,
        'out/examples/mpl_line.mp4'
    )

if __name__ == '__main__':
    _main()





