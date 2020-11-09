from threading import Thread
from queue import Queue, Empty
import time

# TODO: use multiprocessing module instead ?

# http://www.dabeaz.com/generators/Generators.pdf


class PipelineStage(Thread):
    def __init__(self):
        super().__init__()
        self._prev_stage = None
        self.out_queue = Queue()
        self._running = False
        self._processing_time_averager = Averager()
        self._wait_time_averager = Averager()
        self._last_output = None
        self._last_output_timestamp = time.time()

    def subscribe_to(self, prev_stage=None):
        self._prev_stage = prev_stage

    def run(self):
        self._running = True
        while self._running:
            self.loop()
        self._on_stopped()

    def loop(self):
        if self._prev_stage:
            in_queue = self._prev_stage.out_queue
            # Wait for new in_data from subscription
            try:
                in_data = in_queue.get(block=True, timeout=10/1000)
                self._wait_time_averager.add_value(time.time() - self._last_output_timestamp)
            except Empty:
                return None
        else:
            in_data = None
        t1 = time.time()
        result = self._process(in_data)
        t2 = time.time()
        self.out_queue.put(result)
        self._last_output = result
        self._last_output_timestamp = time.time()
        self._processing_time_averager.add_value(t2-t1)

    def stop(self):
        self._running = False
        # _on_stopped() will be called when the while loop in run() exits

    def get_avg_processing_time(self):
        return self._processing_time_averager.get_average()

    def get_avg_wait_time(self):
        return self._wait_time_averager.get_average()

    def get_avg_processing_capacity(self):
        avg_processing = self.get_avg_processing_time()
        avg_wait = self.get_avg_wait_time()
        return avg_processing / (avg_wait + avg_processing)

    def get_last_output(self):
        return self._last_output

    def _process(self, in_data):
        """
        Processes input data and generates an output for this pipeline stage. It's called whenever
        new data from the previous stage arrives.
        :param in_data: input received from the previous stage.
        :return: the output for this pipeline stage. Will be put on the queue and read by the next
        stage, if any.
        """
        raise NotImplemented()

    def _on_stopped(self):
        """
        Called when this pipeline stage is about to be stopped. Perform finalization tasks here.
        """
        pass


class Producer(PipelineStage):
    def _process(self, in_data):
        return self._produce()

    def _produce(self):
        raise NotImplemented()

    def subscribe_to(self, prev_stage=None):
        raise NotImplemented("A Producer can't subscribe to another PipelineStage")


class Consumer(PipelineStage):
    def _process(self, in_data):
        with self.out_queue.mutex:
            self.out_queue.queue.clear()
        return self._consume(in_data)

    def _consume(self, in_data):
        raise NotImplemented()


class Averager:
    def __init__(self):
        self._count = 0
        self._avg = 0

    def add_value(self, value):
        self._avg = ((self._count * self._avg) + value) / (self._count + 1)
        self._count += 1

    def get_average(self):
        return self._avg
