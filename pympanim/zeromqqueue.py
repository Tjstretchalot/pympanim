"""This module allows for rapid replacement of the slow multiprocessing queues
with faster implementations from other libraries without too much development work.
"""

import zmq
import queue
import time
import pytypeutils as tus

class ZeroMQQueue:
    """A queue-like object that uses zmq as the backend. Each side is unidirectional,
    and the direction swaps when you serialize. Typical usecase goes as follows:

        def _my_target(jobq, resq):
            jobq = ZeroMQQueue.deser(jobq)
            resq = ZeroMQQueue.deser(resq)

            while True:
                msg = jobq.get()
                if not msg[0]:
                    break
                resq.put(msg[1] * 2)

        job_queue = ZeroMQQueue.create_send()
        result_queue = ZeroMQQueue.create_recieve()

        proc = Process(target=_my_target, args=(job_queue.serd(), result_queue.serd()))

    Attributes:
        connection (zmq.Socket): the socket we are connected with
        port (int): the port we are attached to
        is_output (bool): if we put to this queue (True) or get from this queue (False)

        last_val (any): the last value retrieved but not returned, used for empty checks
        have_last_val (bool): true if we have last_val, false otherwise

        _last_eagain_spam (float, optional): the last time we printed out that we received
            an eagain error.
    """

    def __init__(self, connection: zmq.Socket, port: int, is_output: bool):
        tus.check(port=(port, int), is_output=(is_output, bool))
        self.connection = connection
        self.port = port
        self.is_output = is_output

        self.last_val = None
        self.have_last_val = False

        self._last_eagain_spam = None

    @classmethod
    def deser(cls, serd):
        """Deserializes the other side of the ZeroMQQueue"""

        port = serd[0]
        is_out = serd[1]

        context = zmq.Context()

        connection = context.socket(zmq.PUSH if is_out else zmq.PULL) # pylint: disable=no-member
        connection.connect(f'tcp://127.0.0.1:{port}')

        return cls(connection, port, is_out)

    @classmethod
    def create_recieve(cls, port=None):
        """Creates a queue which you can get() from. If the port is specified
        this connects to it, otherwise it binds to a random port."""

        context = zmq.Context()
        connection = context.socket(zmq.PULL) # pylint: disable=no-member
        if port is None:
            port = connection.bind_to_random_port('tcp://127.0.0.1')
        else:
            connection.connect(f'tcp://127.0.0.1:{port}')

        return cls(connection, port, False)

    @classmethod
    def create_send(cls, port=None):
        """Creates a queue which you can put() into.  If the port is specified
        this connects to it, otherwise it binds to a random port."""

        context = zmq.Context()
        connection = context.socket(zmq.PUSH) # pylint: disable=no-member
        if port is None:
            port = connection.bind_to_random_port('tcp://127.0.0.1')
        else:
            connection.connect(f'tcp://127.0.0.1:{port}')

        return cls(connection, port, True)

    def _again(self, desc=None):
        """Called when we receive an EAGAIN error"""
        if self._last_eagain_spam is None or time.time() > self._last_eagain_spam + 5:
            print(f'Received EAGAIN response - output full! - (operation: {desc}) - sleeping 1ms')
            self._last_eagain_spam = time.time()
        time.sleep(0.001)

    def get(self, block=True, timeout=None): # pylint: disable=unused-argument
        """Gets the next value from the queue, blocking by default. Timeout is ignored"""
        if self.is_output:
            raise RuntimeError('tried to get from a put-only queue')
        if not block:
            return self.get_nowait()

        if self.have_last_val:
            self.have_last_val = False
            val = self.last_val
            self.last_val = None
            return val

        while True:
            try:
                return self.connection.recv_pyobj()
            except zmq.Again:
                self._again('recv_pyobj')

    def get_nowait(self):
        """Gets the value in the queue if there is one, raises queue.Empty if not"""
        if self.is_output:
            raise RuntimeError('tried to get from a put-only queue')

        if self.have_last_val:
            self.have_last_val = False
            val = self.last_val
            self.last_val = None
            return val

        try:
            return self.connection.recv_pyobj(zmq.NOBLOCK) # pylint: disable=no-member
        except zmq.ZMQError as exc:
            raise queue.Empty from exc

    def put(self, val, block=True, timeout=None): #pylint: disable=unused-argument
        """Puts a value into the queue. Blocks if block=True, otherwise is put_nowait.
        Timeout is ignored"""
        if not self.is_output:
            raise RuntimeError('tried to put into a get-only queue')
        if not block:
            return self.put_nowait(val)

        while True:
            try:
                return self.connection.send_pyobj(val)
            except zmq.Again:
                self._again('send_pyobj')

    def put_nowait(self, val):
        """Puts a value into the queue, nonblocking. Raises queue.Full if not"""
        if not self.is_output:
            raise RuntimeError('tried to put into a get-only queue')

        try:
            self.connection.send_pyobj(val, zmq.NOBLOCK) # pylint: disable=no-member
        except zmq.ZMQError as exc:
            raise queue.Full from exc

    def empty(self):
        """Returns True if the queue is empty, False otherwise"""
        if self.have_last_val:
            return False

        try:
            self.last_val = self.get_nowait()
            self.have_last_val = True
            return False
        except queue.Empty:
            return True

    def serd(self, same_side=False):
        """Creates an object which can be passed to deser to get the opposite end
        of this queue.

        Args:
            same_side (bool): if True we do not change sides, False to change sides
        """
        if same_side:
            side = self.is_output
        else:
            side = not self.is_output
        return (self.port, side)

    def close(self):
        """Explicitly closes this queue"""
        self.connection.close()
