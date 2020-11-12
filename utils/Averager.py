
class Averager:
    def __init__(self):
        self._count = 0
        self._avg = 0

    def add_value(self, value):
        self._avg = ((self._count * self._avg) + value) / (self._count + 1)
        self._count += 1

    def get_average(self):
        return self._avg
