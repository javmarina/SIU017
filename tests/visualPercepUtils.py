import math

import matplotlib.pyplot as plt


def computeRowsCols(N, m, n):
    """
    Calcula el número de filas y columnas necesarias para un subplot
    :param N: número de subfiguras
    :param m: preferencia de filas
    :param n: preferencia de columnas
    :return: número de filas y de columnas (tupla) generado
    """
    # print(N, m, n)
    if m is None:
        m = math.sqrt(N)
        if n is None:
            n = math.ceil(N / m)
        else:
            m = math.ceil(N / n)
    else:
        if n is None:
            n = math.ceil(N / m)
        else:
            m = math.ceil(N / n)
    m, n = max(1, m), max(1, n)
    m, n = math.ceil(m), math.ceil(n)
    # print(m, n)
    return m, n


def showInGrid(imgs, m=None, n=None, title="", subtitles=None, xlims=None, ylims=None):
    """
    Mostrar las imágenes en subplots de la misma figura.
    :param imgs: imágenes a mostrar
    :param m: preferencia de número de filas
    :param n: preferencia de número de columnas
    :param title: título global de la figura
    :param subtitles: subtítulo de cada subfigura (una por imagen)
    :param xlims: límites en el eje X (el orden se ignora para mantener la imagen derecha)
    :param ylims: límites en el eje Y (el orden se ignora para mantener la imagen derecha)
    :return: None
    """
    N = len(imgs)

    m, n = computeRowsCols(N, m, n)
    # print(m,n)
    fig = plt.figure(figsize=(m, n))
    plt.gray()
    for i in range(1, N + 1):  # por cada imagen...
        ax = fig.add_subplot(m, n, i)
        if len(imgs[i - 1].shape) >= 2:
            # Es imagen 2D
            plt.imshow(imgs[i - 1])
            if xlims:
                if xlims[0] < xlims[1]:
                    plt.xlim(xlims[0], xlims[1])
                else:
                    plt.xlim(xlims[1], xlims[0])
            if ylims:
                if ylims[0] < ylims[1]:
                    plt.ylim(ylims[1], ylims[0])
                else:
                    plt.ylim(ylims[0], ylims[1])
        else:
            # Es un vector unidimensional
            plt.plot(imgs[i - 1])
        if subtitles is not None:
            ax.set_title(subtitles[i - 1])

    fig.suptitle(title)
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.show(block=True)
