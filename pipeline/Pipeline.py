
class Pipeline:
    """
    Generic pipeline processing structure. Made of pipeline stages (independent threads) that can process inputs
    as they come and generate outputs.
    This class doesn't modify any subscription. Is up to the caller to manage them.
    """
    def __init__(self, stages):
        self._stages = stages
        self._running = False

    def start(self):
        for stage in reversed(self._stages):
            stage.start()
        self._running = True

    def stop(self):
        for stage in self._stages:
            stage.stop()
            stage.join()
        self._running = False

    def print_all_debug_info(self):
        self.print_avg_processing_times()
        self.print_avg_wait_times()
        self.print_avg_processing_capacity()
        self.print_queue_sizes()

    def print_avg_processing_times(self):
        print("Average processing time for every pipeline stage")
        for i in range(len(self._stages)):
            stage = self._stages[i]
            print(f"Stage {i}, type {type(stage)}:", 1000*stage.get_avg_processing_time(), "ms")

    def print_avg_wait_times(self):
        print("Average wait time for every pipeline stage")
        for i in range(len(self._stages)):
            stage = self._stages[i]
            print(f"Stage {i}, type {type(stage)}:", 1000*stage.get_avg_wait_time(), "ms")

    def print_avg_processing_capacity(self):
        print("Average processing capacity for every pipeline stage")
        for i in range(len(self._stages)):
            stage = self._stages[i]
            print(f"Stage {i}, type {type(stage)}: ", 100*stage.get_avg_processing_capacity(), "%", sep="")

    def print_queue_sizes(self):
        print("Queue sizes for every pipeline stage")
        for i in range(len(self._stages)):
            stage = self._stages[i]
            print(f"Stage {i}, type {type(stage)}:", stage.out_queue.qsize(), "item(s)")

    # def get_stage_by_index(self, index):
    #     return self._stages[index]
    #
    # def get_stage_by_class(self, stage_class):
    #     for stage in self._stages:
    #         if isinstance(stage, stage_class):
    #             return stage
    #     return None


class StraightPipeline(Pipeline):
    """
    Straight pipeline. The graph for this pipeline is a straight list. Every pipeline stage, except the first
    one, is subscribed to the previous stage.
    """
    def __init__(self, stages):
        """
        Creates an straight pipeline.
        :param stages: ordered list of the pipeline stages. The first one is assumed to be a Producer and
        the last one a Consumer.
        """
        super().__init__(stages)
        for i in range(1, len(self._stages)):
            self._stages[i].subscribe_to(self._stages[i - 1])

    def get_consumer_output(self):
        return self._stages[-1].get_last_output()
