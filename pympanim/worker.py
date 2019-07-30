"""This module actually produces a movie from a frame generator. This attempts
to utilize all the computing resources available to it.
"""

import time
import queue
import typing
import traceback
import psutil
import logging
import multiprocessing as mp
from multiprocessing import Process
import pympanim.frame_gen as fg
import pympanim.image_stich as imgst
import pytypeutils as tus
from pympanim.zeromqqueue import ZeroMQQueue

class FrameWorker:
    """Describes something which can send and receive messages to/from the
    main thread in addition to sending images to the image stiching thread. To
    avoid strange rounding issues, a frame worker has a knowledge of the frames
    per second and thus corresponds using frame numbers rather than time, since
    frame numbers are integers.

    Attributes:
        img_queue (Queue): the queue we push images to
        rec_queue (Queue): the queue we receive messages from
        send_queue (Queue): the queue we push responses to the main thread from

        frame_gen (FrameGenerator): the thing which actually generates frames.
        ms_per_frame (float): the number of milliseconds per frame
    """
    def __init__(self, img_queue, rec_queue, send_queue, frame_gen: fg.FrameGenerator,
                 ms_per_frame: float):
        self.img_queue = img_queue
        self.rec_queue = rec_queue
        self.send_queue = send_queue
        self.frame_gen = frame_gen
        self.ms_per_frame = ms_per_frame

    def do_all(self):
        """This is meant to be the function that is invoked immediately after
        initialization. This processes the receive queue until completion.
        """
        self.frame_gen.start()

        while True:
            msg = self.rec_queue.get()
            if msg[0] == 'sync':
                self.send_queue.put(('sync', time.time()))
                continue
            if msg[0] == 'finish':
                break
            if msg[0] != 'img':
                raise ValueError(f'strange msg: {msg}')

            frame_num = msg[1]
            time_ms = self.ms_per_frame * frame_num
            rawimg = self.frame_gen.generate_at(time_ms)
            self.img_queue.put((frame_num, rawimg))
            self.send_queue.put(('post', frame_num))
            rawimg = None

        self.frame_gen.finish()

        self.img_queue.close()
        self.rec_queue.close()
        self.send_queue.close()

def frame_worker_target(img_queue, rec_queue, send_queue, frame_gen,
                        ms_per_frame, error_file):
    """Creates a frame worker with the given arguments and then runs it.
    The queues are assumed to be ZeroMQ queues which are serialized.

    The error file is not passed to the FrameWorker, but is instead where
    errors are stored if one occurs.
    """
    img_queue = ZeroMQQueue.deser(img_queue)
    rec_queue = ZeroMQQueue.deser(rec_queue)
    send_queue = ZeroMQQueue.deser(send_queue)

    try:
        FrameWorker(img_queue, rec_queue, send_queue, frame_gen,
                    ms_per_frame).do_all()
    except:
        traceback.print_exc()
        with open(error_file, 'w') as outfile:
            traceback.print_exc(file=outfile)
        raise

class FrameWorkerConnection:
    """An instance in the main thread that describes a connection with a frame worker
    Attributes:
        proc (Process): the actual child process
        img_queue (queue): the queue the frame worker sends images to us with
        send_queue (queue): the queue we send the frame worker messages with
        ack_queue (queue): the queue we receive messages from the frame worker from
        awaiting_sync (bool): True if we are awaiting a sync message, false otherwise

        in_queue (int): the number of frames the worker still has to do
        num_since_sync (int): the number of frames sent since the last sync
        last_frame (int): the last frame the worker was asked to process
    """

    def __init__(self, proc: Process, img_queue, send_queue, ack_queue):
        self.proc = proc
        self.img_queue = img_queue
        self.send_queue = send_queue
        self.ack_queue = ack_queue
        self.awaiting_sync = False
        self.in_queue = 0
        self.num_since_sync = 0
        self.last_frame = -1

    def handle_post_frame(self, msg):
        """Invoked internally after the worker acknowledges he has
        completed a frame"""
        self.in_queue -= 1

    def handle_sync(self, msg):
        """Invoked internally after the worker responds to a sync request"""
        sync_time = time.time() - msg[1]
        if sync_time > 5:
            print(f'[FrameWorkerConnection] took a long time to sync ({sync_time:.3f} s)')
        self.awaiting_sync = False
        self.num_since_sync = 0

    def handle_ack(self, msg):
        """Invoked internally for messages from the worker"""
        if msg[0] == 'post':
            self.handle_post_frame(msg)
        elif msg[0] == 'sync':
            self.handle_sync(msg)
        else:
            raise ValueError(f'unknown ack: {msg}')

    def check_ack_queue(self):
        """Checks the queue that the worker uses to talk to us"""
        try:
            while True:
                ack = self.ack_queue.get_nowait()
                self.handle_ack(ack)
        except queue.Empty:
            pass

    def start_sync(self):
        """Starts the syncing process"""
        self.send_queue.put(('sync', time.time()))
        self.awaiting_sync = True

    def check_sync(self):
        """Checks if the syncing process is complete"""
        if not self.awaiting_sync:
            return True
        self.check_ack_queue()
        return not self.awaiting_sync

    def sync(self):
        """Waits for this worker to catch up"""
        self.start_sync()
        while self.awaiting_sync:
            resp = self.ack_queue.get()
            self.handle_ack(resp)
        return time.time() - resp[1]

    def start_finish(self):
        """Starts the finish process"""
        self.send_queue.put(('finish',))

    def check_finish(self):
        """Checks if the worker has shutdown yet"""
        return not self.proc.is_alive()

    def wait_finish(self):
        """Waits for finish process to complete"""
        self.proc.join()

    def finish(self):
        """Cleanly shutdowns the worker"""
        self.start_finish()
        self.wait_finish()

    def send(self, frame_num):
        """Notifies this worker that it should render the specified frame number"""
        self.send_queue.put(('img', frame_num))
        self.in_queue += 1
        self.num_since_sync += 1
        self.last_frame = frame_num

    def offer(self, frame_num, target_in_queue) -> bool:
        """If this worker has fewer than target_in_queue items in its queue,
        then we send the specified frame numebr to the worker and return true.
        Otherwise, we return false.
        """
        if self.in_queue < target_in_queue:
            self.send(frame_num)
            return True
        return False

    def close(self):
        """Closes all queues"""
        self.img_queue.close()
        self.send_queue.close()
        self.ack_queue.close()

class PerformanceSettings:
    """These are the settings that are used for performance. These are modified
    by runtime dynamics so when repeatedly running the same  or significantly
    similar videos, it can be helpful to store these between runs. These are
    all primitives, so you can pickle this or use __dict__ to dump via json.

    Attributes:

        TRAINED:

        frames_per_sync (int): the number of frames before we ask workers to
            sync. Larger numbers give less feedback about how the workers
            are doing but waste less time. this is typically annealed
            upward to max_frames_per_sync.
        num_workers (int): the number of worker threads. the key variable.
            Defaults to 1/3 the number of physical cores.
        frame_batch_amount (int): the number of sequential frames sent to
            each worker. Sequential frames are sometimes faster to calculate.

        UNTRAINED:

        window_size (float): the number of seconds that a performance window
            lasts. Must not be 0.
        perf_delay (float): the number of seconds before we expect a parameter
            change to have some effect on performance. Can be 0
        max_workers (int): the maximum number of threads generating images,
            defaults to 2/3 number of physical cores
        worker_queue_size (int): the number of jobs we try to maintain in the
            queue for each worker.
        work_per_dispatch (int): the amount of work we do between scanning the
            worker queue to see if we need to dispatch jobs. Increasing the
            queue size allows increasing this
        spawn_worker_threshold_low (float): the percentage of frames received
            that are processed by the image thread in a given amount of time
            that triggers spawning another worker when we have have appreciably
            below the ooo_balance number of frames.

            For example, 0.8 means that if we processed 80% or more of the frames
            we received in a given window in that same window, we should spawn
            another worker.
        spawn_worker_threshold_high (float): analgous to the previous, except
            this is the threshold used when we have an acceptable number of
            out of order frames. For example, 1.2 means if we processed 120%
            of the frames received in a given window, we should spawn naother
            thread in anticipation of the number of ooo frames falling
        kill_worker_threshold_low (float): the percentage of frames received
            that are processed by the image thread in a given amount of time
            that triggers killing a worker when we have appreciably below the
            ooo_balance number of frames.

            For example, 0.6 means that even if we don't have enough ooo
            frames, but we only processed 60% of the received images in a given
            period, we should spawn another worker in anticipation of the
            number of ooo frames growing excessively.
        kill_worker_threshold_high (float): analagous to
            kill_worker_threshold_high except when we have sufficiently many
            out of order frames.

            For example, 0.8 means that if we have enough ooo frames but we
            only processed 80% of the incoming images in a given window, then
            we should kill a worker to prevent getting too many more ooo
            frames.

        ooo_balance (int): the minimum number of frames that puts us into
            the high threshold for the above parameters.
        ooo_cap (int): the number of out of order frames that causes us to
            stop releasing jobs until we fall below. It should be much
            larger than the balance since killing workers is a much more
            efficient way to reduce out of order frames
        ooo_error (int): the number of out of order frames that causes
            us to error.
        min_frames_per_sync (int): the minimum number of frames per sync,
            regardless of how (non)confident we are that we have achieved
            a balance of dispatching jobs and working
        max_frames_per_sync (int): the maximum number of frames per sync,
            regardless of how confident we are that we have achieved a
            balance of dispatching jobs and working.


        frame_batch_min_improvement (float): the minimum performance improvemnt
            for us to increment the frame batch amount. for example, 1.1 means
            we will increase the frame batch size up to the point where doing
            so gives us less than 10% improvement in performance. Use 1 + eps
            to increase if theres any performance.
        frame_batch_max_badness (float): the maximum detected performance
            reduction before we reduce the frame batch amount. For example,
            0.9 means that if we can reduce the frame batch size and keep 90%
            of performance than we should do so
        frame_batch_dyn_min_decay_time (float): the number of seconds before
            a dynamic minimum decays by 1, meaning we retry a minimum
        frame_batch_dyn_max_decay_time (float): the number of seconds before
            a dynamic maximum decays by 1, meaning we retry a maximum
        frame_batch_min (int): the minimum frame batch size
        frame_batch_max (int): the maximum frame batch size. it should be noted
            that in general more frame batches means more out of order frames,
            so increasing this should involve increasing ooo_balance and
            ooo_cap.
    """
    def __init__(
            self, frames_per_sync=10, num_workers=None,
            frame_batch_amount=2,
            window_size=15.0,
            perf_delay=2.5,
            max_workers=None,
            worker_queue_size=5,
            work_per_dispatch=4,
            spawn_worker_threshold_low=0.8,
            spawn_worker_threshold_high=1.2,
            kill_worker_threshold_low=0.5,
            kill_worker_threshold_high=0.8,
            ooo_balance=100, # this is 829mb of 1920x1080 images
            ooo_cap=500, # this is 4.15 gb of said images in memory
            ooo_error=5000, # 41.5 gb of said images in memory
            min_frames_per_sync=100,
            max_frames_per_sync=1500,
            frame_batch_min_improvement=1.05,
            frame_batch_max_badness=1.0,
            frame_batch_dyn_min_decay_time=120.0,
            frame_batch_dyn_max_decay_time=120.0,
            frame_batch_min=1,
            frame_batch_max=10
        ):
        if num_workers is None:
            num_workers = psutil.cpu_count(logical=False) // 3
        if max_workers is None:
            max_workers = max(
                num_workers,
                (psutil.cpu_count(logical=False) // 3) * 2
            )

        tus.check(
            frames_per_sync=(frames_per_sync, int),
            num_workers=(num_workers, int),
            frame_batch_amount=(frame_batch_amount, int),
            window_size=(window_size, float),
            perf_delay=(perf_delay, float),
            max_workers=(max_workers, int),
            worker_queue_size=(worker_queue_size, int),
            work_per_dispatch=(work_per_dispatch, int),
            spawn_worker_threshold_low=(spawn_worker_threshold_low, float),
            spawn_worker_threshold_high=(spawn_worker_threshold_high, float),
            kill_worker_threshold_low=(kill_worker_threshold_low, float),
            kill_worker_threshold_high=(kill_worker_threshold_high, float),
            ooo_balance=(ooo_balance, int),
            ooo_cap=(ooo_cap, int),
            ooo_error=(ooo_error, int),
            min_frames_per_sync=(min_frames_per_sync, int),
            max_frames_per_sync=(max_frames_per_sync, int),
            frame_batch_min_improvement=(frame_batch_min_improvement, float),
            frame_batch_max_badness=(frame_batch_max_badness, float),
            frame_batch_dyn_min_decay_time=(frame_batch_dyn_min_decay_time, float),
            frame_batch_dyn_max_decay_time=(frame_batch_dyn_max_decay_time, float),
            frame_batch_min=(frame_batch_min, int),
            frame_batch_max=(frame_batch_max, int)
        )

        self.frames_per_sync = frames_per_sync
        self.num_workers = num_workers
        self.frame_batch_amount = frame_batch_amount
        self.window_size = window_size
        self.perf_delay = perf_delay
        self.max_workers = max_workers
        self.worker_queue_size = worker_queue_size
        self.work_per_dispatch = work_per_dispatch
        self.spawn_worker_threshold_low = spawn_worker_threshold_low
        self.spawn_worker_threshold_high = spawn_worker_threshold_high
        self.kill_worker_threshold_low = kill_worker_threshold_low
        self.kill_worker_threshold_high = kill_worker_threshold_high
        self.ooo_balance = ooo_balance
        self.ooo_cap = ooo_cap
        self.ooo_error = ooo_error
        self.min_frames_per_sync = min_frames_per_sync
        self.max_frames_per_sync = max_frames_per_sync
        self.frame_batch_min_improvement = frame_batch_min_improvement
        self.frame_batch_max_badness = frame_batch_max_badness
        self.frame_batch_dyn_min_decay_time = frame_batch_dyn_min_decay_time
        self.frame_batch_dyn_max_decay_time = frame_batch_dyn_max_decay_time
        self.frame_batch_min = frame_batch_min
        self.frame_batch_max = frame_batch_max

def _spawn_worker(frame_gen, ms_per_frame, i):
    img_queue = ZeroMQQueue.create_recieve()
    send_queue = ZeroMQQueue.create_send()
    ack_queue = ZeroMQQueue.create_recieve()
    proc = Process(
        target=frame_worker_target,
        args=(img_queue.serd(), send_queue.serd(), ack_queue.serd(), frame_gen,
              ms_per_frame, f'worker_{i}_error.log')
    )
    proc.start()
    return FrameWorkerConnection(proc, img_queue, send_queue, ack_queue)


def produce(frame_gen: fg.FrameGenerator, fps: float,
            dpi: typing.Union[int, float], bitrate: typing.Union[int, float],
            outfile: str,
            settings: PerformanceSettings = None, time_per_print: float = 15.0,
            logger: logging.Logger = None) -> PerformanceSettings:
    """Produces a video with the given frame rate (specified as milliseconds
    per frame), using the given performance settings. If the performance
    settings are not provided, reasonable defaults are used. Returns the final
    performance settings, which may have changed over the course of video
    production. It may improve performance to reseed with the settings that
    this ended with.

    Arguments:
        frame_gen (FrameGenerator): the thing that generates frames that you
            want to turn into a movie.
        fps (int, float): the number of frames per second
        dpi (int, float): the number of pixels per inch
        outfile (str): the path to the mp4 file where you want to save
            the movie to. Should not already exist. The directory up
            to this point will be auto-generated.
        settings (PerformanceSettings, optional): the performance settings
            to use. For longer videos, more aggressive optimization might
            be appropriate, while for shorter videos more conservative
            optimization might be appropriate.
        time_per_print (float): the number of seconds between us logging
            performance / progress information
        logger (Logger): the logger to use for printing progress information
    """

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    if settings is None:
        settings = PerformanceSettings()
    if logger is None:
        logger = logging.getLogger('pympanim.worker')
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(
            format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p')

    ms_per_frame = 1000 / fps
    num_frames = int(frame_gen.duration / ms_per_frame)
    logger.info('Settings: %0.1f seconds; %d frames at %d fps with %d workers...',
                frame_gen.duration / 1000, num_frames, fps, settings.num_workers)

    workers = []
    paused_workers = []
    stopping_workers = [] # closed when we process their last frame

    perf = imgst.ISRunningAveragePerfHandler(settings.window_size)
    isticher = imgst.ImageSticher(frame_gen.frame_size, dpi, bitrate, fps,
                                  outfile, settings.ooo_error)
    isticher.perfs.append(perf)

    for i in range(settings.num_workers):
        worker = _spawn_worker(frame_gen, ms_per_frame, i)
        isticher.register_queue(worker.img_queue)
        workers.append(worker)

    worker_counter = settings.num_workers

    for worker in workers:
        worker.start_sync()
    isticher.start()

    all_synced = False
    while not all_synced:
        all_synced = True
        for worker in workers:
            if not worker.check_sync():
                all_synced = False
        time.sleep(0.001)

    old_perf = None
    cur_optim = None # magical string values
    frame_batch_dyn_min = settings.frame_batch_min
    frame_batch_dyn_max = settings.frame_batch_max
    frame_batch_min_next_decay = float('inf')
    frame_batch_max_next_decay = float('inf')
    next_optim = time.time() + settings.perf_delay + settings.window_size
    next_progress = time.time() + max(settings.perf_delay + settings.window_size, time_per_print)


    cur_frame = 0
    syncing = False

    while cur_frame < num_frames:
        if not syncing:
            frames_per_worker_since_sync = 0
            for worker in workers:
                worker.check_ack_queue()
                while worker.offer(cur_frame, settings.worker_queue_size):
                    cur_frame += 1
                    frames_per_worker_since_sync = max(
                        frames_per_worker_since_sync, worker.num_since_sync)
                    if cur_frame >= num_frames:
                        break
                    for i in range(settings.frame_batch_amount - 1):
                        worker.send(cur_frame)
                        cur_frame += 1
                        frames_per_worker_since_sync = max(
                            frames_per_worker_since_sync, worker.num_since_sync)
                        if cur_frame >= num_frames:
                            break
                    if cur_frame >= num_frames:
                        break
                if cur_frame >= num_frames:
                    break
            if cur_frame >= num_frames:
                break

            if frames_per_worker_since_sync > settings.frames_per_sync:
                for worker in workers:
                    worker.start_sync()
                syncing = True
        else:
            syncing = False
            for worker in workers:
                if not worker.check_sync():
                    syncing = True
                    break

        for i in range(settings.work_per_dispatch):
            isticher.do_work()

        while len(isticher.ooo_frames) > settings.ooo_cap:
            isticher.do_work()

        for i in range(len(stopping_workers) - 1, 0, -1):
            worker = stopping_workers[i]
            if worker.check_finish() and isticher.next_frame > worker.last_frame:
                worker.check_sync() # cleanup just in case
                isticher.remove_queue(worker.img_queue)
                worker.close()
                stopping_workers.pop(i)

        thetime = time.time()
        if thetime >= next_progress:
            next_progress = thetime + time_per_print
            recpsec, procpsec = perf.mean()
            frames_to_proc = num_frames - isticher.next_frame
            time_left_sec = frames_to_proc / procpsec if procpsec > 0 else float('inf')
            logger.info('[%0.1f secs remaining] Generating %0.2f images/sec and ' # pylint: disable=logging-not-lazy
                        + 'processing %0.2f images/sec', time_left_sec,
                        recpsec, procpsec)

        if thetime >= next_optim:
            next_optim = thetime + settings.perf_delay + settings.window_size
            if frame_batch_min_next_decay < thetime:
                frame_batch_dyn_min -= 1
                frame_batch_min_next_decay = (
                    float('inf') if frame_batch_dyn_min <= settings.frame_batch_min
                    else thetime + settings.frame_batch_dyn_min_decay_time
                )
            if frame_batch_max_next_decay < thetime:
                frame_batch_dyn_max += 1
                frame_batch_max_next_decay = (
                    float('inf') if frame_batch_dyn_max >= settings.frame_batch_max
                    else thetime + settings.frame_batch_dyn_max_decay_time
                )

            recpsec, procpsec = perf.mean()
            if old_perf is not None and cur_optim is not None:
                oldrecpsec, oldprocpsec = old_perf # pylint: disable=unpacking-non-sequence, unused-variable

                if cur_optim == 'reduce_frame_batch_amount':
                    relative_performance = oldprocpsec / procpsec # prob <1
                    if relative_performance > settings.frame_batch_max_badness:
                        # keep the change
                        logger.debug(
                            'found better setting: frame_batch_amount=%d (rel performance: %0.3f)',
                            settings.frame_batch_amount, relative_performance)
                        frame_batch_dyn_max = settings.frame_batch_amount
                        frame_batch_max_next_decay = (
                            thetime + settings.frame_batch_dyn_max_decay_time
                        )
                    else:
                        # revert the change
                        # we're evil scientists so we dont report null results
                        settings.frame_batch_amount += 1
                        frame_batch_dyn_min = settings.frame_batch_amount
                        frame_batch_min_next_decay = (
                            thetime + settings.frame_batch_dyn_min_decay_time
                        )
                elif cur_optim == 'increase_frame_batch_amount':
                    relative_performance = oldprocpsec / procpsec # prob >1
                    if relative_performance > settings.frame_batch_min_improvement:
                        # keep the change
                        logger.debug(
                            'found better setting: frame_batch_amount=%d (rel performance: %0.3f)',
                            settings.frame_batch_amount, relative_performance)
                        frame_batch_dyn_min = settings.frame_batch_amount
                        frame_batch_min_next_decay = (
                            thetime + settings.frame_batch_dyn_min_decay_time
                        )
                    else:
                        # revert the change
                        # we're evil scientists so we dont report null results
                        settings.frame_batch_amount -= 1
                        frame_batch_dyn_max = settings.frame_batch_amount
                        frame_batch_max_next_decay = (
                            thetime + settings.frame_batch_dyn_max_decay_time
                        )
                else:
                    raise RuntimeError(f'unknown cur_optim = {cur_optim}')

                cur_optim = None

            perc_rec_proc = procpsec / recpsec
            reason_str = (f'(processing {perc_rec_proc:.3f} images for every '
                          + f'image generated, have {len(isticher.ooo_frames)} '
                          + 'frames awaiting processing)')

            threshold_spawn, threshold_kill = (
                (settings.spawn_worker_threshold_low,
                 settings.kill_worker_threshold_low)
                if len(isticher.ooo_frames) < settings.ooo_balance
                else (settings.spawn_worker_threshold_high,
                      settings.kill_worker_threshold_high)
            )

            if (perc_rec_proc > threshold_spawn
                    and settings.num_workers < settings.max_workers):
                settings.num_workers += 1
                if settings.frames_per_sync > settings.min_frames_per_sync:
                    settings.frames_per_sync -= 1
                if paused_workers:
                    unpaused = paused_workers.pop()
                    workers.append(unpaused)
                    logger.debug('Unpaused a worker %s', reason_str)
                else:
                    worker = _spawn_worker(frame_gen, ms_per_frame, worker_counter)
                    isticher.register_queue(worker.img_queue)
                    workers.append(worker)
                    worker_counter += 1
                    logger.debug('Spawned a worker %s', reason_str)
            elif (perc_rec_proc < threshold_kill
                    and settings.num_workers > 1):
                settings.num_workers -= 1
                if settings.frames_per_sync > settings.min_frames_per_sync:
                    settings.frames_per_sync -= 1
                settings.frames_per_sync -= 1
                if not paused_workers:
                    paused = workers.pop()
                    paused_workers.append(paused)
                    logger.debug('Paused a worker %s', reason_str)
                else:
                    paused = workers.pop()
                    killed = paused_workers.pop()
                    paused_workers.append(paused)
                    stopping_workers.append(killed)
                    killed.start_finish()
                    logger.debug('Killed a worker %s', reason_str)
            elif settings.frames_per_sync < settings.max_frames_per_sync:
                settings.frames_per_sync += 1

            want_reduce_frame_batch = perc_rec_proc < 1
            # if we have processed fewer than we have received it's not as
            # important that we optimize image generation
            can_reduce_frame_batch = (
                settings.frame_batch_amount > frame_batch_dyn_min
            )
            can_increase_frame_batch = (
                settings.frame_batch_amount < frame_batch_dyn_max
            )

            if ((want_reduce_frame_batch or not can_increase_frame_batch)
                    and can_reduce_frame_batch):
                cur_optim = 'reduce_frame_batch_amount'
                settings.frame_batch_amount -= 1
            elif can_increase_frame_batch:
                cur_optim = 'increase_frame_batch_amount'
                settings.frame_batch_amount += 1


            old_perf = (recpsec, procpsec)


    logger.debug('Shutting down workers...')
    workers.extend(paused_workers)
    paused_workers = []
    for worker in workers:
        worker.start_finish()
    workers.extend(stopping_workers)
    stopping_workers = []

    all_finished = False
    while not all_finished:
        all_finished = not isticher.do_work()
        if not all_finished:
            for worker in workers:
                if not worker.check_finish():
                    all_finished = False
                    break
        if not all_finished:
            for worker in stopping_workers:
                if not worker.check_finish():
                    all_finished = False
                    break

    logger.debug('All workers shut down, processing remaining frames...')
    while isticher.next_frame < num_frames:
        if not isticher.do_work():
            time.sleep(0.001)

    isticher.finish()
    for worker in workers:
        worker.check_sync() # just in case we leaked one
        worker.close()
    logger.info('Finished')
    return settings
