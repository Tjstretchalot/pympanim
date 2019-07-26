"""Various utility functions that don't belong anywhere else"""

import typing

UNITS = (
    (('ms', 'millisecond', 'milliseconds'), 1),
    (('s', 'sec', 'secs', 'second', 'seconds'), 1000),
    (('m', 'min', 'mins', 'minute', 'minutes'), 60 * 1000),
    (('h', 'hr', 'hour', 'hours'), 60 * 60 * 1000),
    (('d', 'dy', 'day', 'days'), 24 * 60 * 60 * 1000)
)
UNITS_LOOKUP = dict()
for names, num_ms in UNITS:
    for nm in names:
        UNITS_LOOKUP[nm] = num_ms

def find_child(ends_arr: typing.List[float],
               time: float,
               hint: int = 0) -> typing.Tuple[int, float]:
    """This function is used when you have a list of times when things end
    (ends_arr), and you are trying to figure out inside which of these
    things time belongs. This is a linear search that breaks the search into
    two based on the hint.

    This assumes that ends_arr is ordered.

    Returns:
        (int): the index in ends_arr for the interval that contains the given
            time.
        (float): the time relative to the child, i.e, the amount of time after
            the end of the previous child
    """
    if hint != 0:
        if ends_arr[hint - 1] <= time < ends_arr[hint]:
            return hint, time - ends_arr[hint - 1]

        if time >= ends_arr[hint]:
            for i in range(hint + 1, len(ends_arr)):
                if time < ends_arr[i]:
                    return i, time - ends_arr[i - 1]

    last = 0
    for i, etime in enumerate(ends_arr):
        if time < etime:
            return i, time - last
        last = etime
    if time == last:
        return len(ends_arr) - 1, 0
    raise ValueError(f'time={time}  not in [0, {ends_arr[-1]}]'
                        + f' or {ends_arr} unsorted')
