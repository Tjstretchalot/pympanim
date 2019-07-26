# Multiprocessed Animations

## Warning

This is currently in pre-release. During this time it may not be available on
PIP and documentation may not be complete or accurate. Development will move
quickly.

## Summary

This library helps building movies in python from images. Specifically, it
intends to allow multiple threads to produce images which are stiched together
into movies with ffmpeg.

## Use-cases

This can be used to multithread the image creation process when using
matplotlib animations. FFMpeg will use multithreading to encode the images into
a video, however producing the images themselves is a bottleneck if you have
many cores available.

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

## Usage

## Examples

The examples/ folder has the sourcecode for the following examples:

```
python3 -m examples.redsquare
```

Produces
