"""Simple utility module that provides documentation for an easing
for the purpose of this module is, and provides a few implementations
for that easing. It is recommended that libraries use the pytweening
library for additional easings.
"""

class Easing:
    """The interface for easings. Things that accept easings should work for
    any callable object that acts like an Easing."""

    def __call__(self, num: float, **kwargs) -> float:
        """Accepts a number between 0 and 1 and returns a number in the
        same range.

        A continuous function f : [0, 1] -> [0, 1] is said to be an easing.
        An easing f is said to be a proper easing if f(0) = 0 and f(1) = 1

        Args:
            num (float): The "uneased" percentage

        Keyword-Arguments:
            Arguments specific to the particular easing.

        Returns:
            eased (float): the "eased" percentage
        """
        raise NotImplementedError

def freeze(n, at=0): # pylint: disable=invalid-name
    """This is the constant function f(n) = at for all n in [0, 1].

    This is **not** a proper easing.
    """
    return at

def squeeze(n, amt=0.2): # pylint: disable=invalid-name
    """Squeezes the n so that instead of going from 0 to 1 in unit time, it goes
    to 0 to 1 in (1-amt) time with padding on both sides.

    This is a proper easing.
    """
    if n < amt:
        return 0
    if n >= 1 - amt:
        return 1
    return (n - amt) / (1 - (amt*2))

def doublespeed(n): # pylint: disable=invalid-name
    """A non-symmetric easing which moves linearly from 0 to 1 in first 0.5
    and then stays constant for last 0.5

    This is a proper easing.
    """

    return n * 2 if n < 0.5 else 1

def smoothstep(n): # pylint: disable=invalid-name
    """A symmetric easing that starts and ends with first derivative 0.

    This is a proper easing.

    Args:
        n (float): a value between 0 and 1
    Returns:
        n (float): a smoothed value between 0 and 1
    """

    return n * n * (3.0 - 2.0 * n)

def smootheststep(n): # pylint: disable=invalid-name
    """A symmetric easing that starts and ends with first, second, and third derivative 0.

    This is a proper easing.

    Args:
        n (float): a value between 0 and 1
    Returns:
        n (float): a smoothed value between 0 and 1
    """
    return n * n * n * n * (35 + n * (-84 + n * (70 + n * -20)))
