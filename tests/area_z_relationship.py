import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    list = None
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
        v = np.polyfit(x, np.log(y), 1)
        X_test = np.array(x)
        Y_pred = np.exp(v[1])*np.exp(X_test*v[0])

        plt.plot(zs, areas)
        plt.plot(X_test, Y_pred)
        plt.show()
