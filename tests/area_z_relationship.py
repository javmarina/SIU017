import pickle

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

if __name__ == "__main__":
    with open("area_z.p", "rb") as f:
        list = pickle.load(f)

    if list is not None:
        areas = []
        zs = []
        for area, z in list:
            areas.append(area)
            zs.append(z)

        x = zs
        y = areas
        y_filtered = savgol_filter(y, window_length=51, polyorder=3)

        plt.plot(x, y, '.', markersize=1.5, label="Valores experimentales")
        plt.plot(x, y_filtered, label="Filtro de Savitzky–Golay")
        plt.legend()
        plt.grid()
        plt.xlabel("Posición Z (m)")
        plt.ylabel("Área en píxeles")
        plt.gcf().tight_layout()
        plt.show()
