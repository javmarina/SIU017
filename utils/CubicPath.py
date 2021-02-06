
class CubicPath:
    """
    Cubic path that smoothly interpolates between two points (x1,y1) and (x2,y2).
    Outside the interpolated region, the curve is flat.
    """

    def __init__(self, x1, y1, x2, y2):
        if x1 > x2:
            x2, x1 = x1, x2
            y2, y1 = y1, y2
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2

        # Polynomial coefficients
        delta_A = self._x2 - self._x1
        delta_v = self._y2 - self._y1
        self._a = -2 * delta_v / (delta_A ** 3)
        self._b = 3 * delta_v / (delta_A ** 2)
        self._c = 0
        self._d = y1

    def __call__(self, x):
        if x <= self._x1:
            return self._y1
        elif x >= self._x2:
            return self._y2
        else:
            return self._a * (x - self._x1) ** 3 + self._b * (x - self._x1) ** 2 \
                   + self._c * (x - self._x1) + self._d


# Test
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    path = CubicPath(x1=2, y1=0.25, x2=8, y2=0.1)
    x = np.linspace(0, 10, 1000)
    y = []
    for x_ in x:
        y.append(path(x_))

    plt.plot(x,y)
    plt.xlabel("Posición estimada en Z (m)")
    plt.ylabel("Velocidad en Z (m/s)")
    plt.grid()
    plt.gcf().tight_layout()
    plt.show(block=True)
