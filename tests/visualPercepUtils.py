import numpy as np
import math
import matplotlib.pyplot as plt

'''
Este archivo contiene utilidades para mostrar por pantalla los resultados (con matplotlib)
He intentado documentar todas las funciones
'''


class Rect:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

def computeRowsCols(N, m, n):
    '''
    Calcula el número de filas y columnas necesarias para un subplot
    :param N: número de subfiguras
    :param m: preferencia de filas
    :param n: preferencia de columnas
    :return: número de filas y de columnas (tupla) generado
    '''
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


# Unused
def showInFigs(imgs, title, nFig=None, bDisplay=False):
    '''
    Mostrar en distintas figuras las imagenes proporcionadas. Todas las figuras comparten el mismo título.
    :param imgs: imágenes a mostrar
    :param title: título de todas las figuras
    :param nFig: número de figuras ya creadas (ID de la última figura). Si no se especifica, modificará figuras ya visibles.
    :param bDisplay: ??
    :return: identificador de la última figura mostrada.
    '''
    # open all images in separate figures without user interation
    i = 0 if nFig is None else nFig + 1
    for im in imgs:
        # print(i)
        plt.figure(i)
        i += 1
        plt.imshow(im, cmap='gray', interpolation=None)  # , aspect=1/1.5)#, vmin=0,vmax=255)
        plt.title(title)
    if bDisplay:
        plt.showb(block=True)
    return i


def showInGrid(imgs, m=None, n=None, title="", subtitles=None, xlims=None, ylims=None):
    '''
    Mostrar las imágenes en subplots de la misma figura.
    :param imgs: imágenes a mostrar
    :param m: preferencia de número de filas
    :param n: preferencia de número de columnas
    :param title: título global de la figura
    :param subtitles: subtítulo de cada subfigura (una por imagen)
    :param xlims: límites en el eje X (el orden se ignora para mantener la imagen derecha)
    :param ylims: límites en el eje Y (el orden se ignora para mantener la imagen derecha)
    :return: None
    '''
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
    mng.window.state('zoomed')
    plt.show(block=True)


def histImg(im):
    '''
    Calcula el histograma de una imagen y lo devuelve como un vector de 256 componentes
    :param im: imagen de la que se quiere calcular el histograma
    :return: vector histograma de la imagen
    '''
    return np.histogram(im.flatten(), 256)[0]


def showPlusInfo(data):
    '''
    Mostrar data en una figura. Bloquea la ejecución del código hasta que se cierra la figura.
    :param data: datos a mostrar en figura.
    :return: None
    '''
    plt.plot(data)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show(block=True)


def showImgsPlusHists(im, im2, title=""):
    '''
    Mostrar dos imágenes junto con sus histogramas
    :param im: primera imagen
    :param im2: segunda imagen
    :param title: título de la figura
    :return: None
    '''
    hists = [histImg(im), histImg(im2)]
    # print(im2.shape, hists[0].shape)
    showInGrid([im, im2] + hists, title=title)
    # alternative possibilities:
    # showInGrid(imgs)
    # showInGrid(hists)
    # showInGrid([im, im2] + hists)
    # showInGrid((imgs[0],hists[0],imgs[1],hists[1]))


# showInGrid(imgs + hists)
# showInGrid((imgs[0],hists[0],imgs[1],hists[1]))
# plt.plot(cdf)
# plt.show()


def pil2np(in_pil):
    '''
    Convertir imágenes en formato PIL (librería Pillow) a formato numpy
    :param in_pil: imágenes de entrada en formato PIL
    :return: imágenes de salida en formato numpy
    '''
    imgs = []
    for im_pil in in_pil:
        print(im_pil.size)
        imgs.extend([np.array(im_pil)])

    return imgs


def displayHoughPeaks(h, peaks, angles, dists, theta, rho):
    nThetas = len(theta)
    rangeThetas = theta[-1] - theta[0]
    slope = nThetas / rangeThetas
    plt.axis('off')
    plt.imshow(h, cmap='jet')
    for peak, angle, dist in zip(peaks, angles, dists):
        print("peak", peak, "at angle", np.rad2deg(angle), "and distance ", dist)
        # y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        # y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
        # ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
        plt.plot(slope * (angle - theta[0]) + 1, dist - rho[0], 'rs',
                 markersize=0.1 * peak)  # size proportional to peak value
    # plt.show()


def showImWithColorMap(im, cmap=None, block=True):
    '''
    Mostrar una imagen con un mapa de color específico.
    :param im: imagen a mostrar. Si es en niveles de grises, utiliza el mapa de color 'cmap'. Para mostrarlo en grayscale,
    usar cmap='grayscale'.
    :param cmap: mapa de color a utilizar. Solo tiene efecto si la imagen es en niveles de gris (ignorado para RGB o RGBA).
    Para mostrar las imagenes en escala de grises sin colores falsos, usar cmap='grayscale'
    :return:
    '''
    plt.imshow(im, cmap=cmap)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show(block=block)
