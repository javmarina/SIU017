
class LowPassFilter:
    def __init__(self, beta, initial_value=0):
        self._value = initial_value
        self._beta = beta

    def add_sample(self, sample):
        self._value += self._beta*(sample-self._value)

    def get_value(self):
        return self._value
