import numpy as np
from scipy import ndimage
from enstools.core import check_arguments
from enstools.interpolation import downsize


@check_arguments(shape={"im": (0, 0)})
def embed_image(im, fac=None, sRow=None, sCol=None):
    """
    embed an array into one that has multiples of fac as sizes; fill missing rows/columns with zeros

    Parameters
    ----------
    im : xarray.DataArray or np.ndarray
            data array, 2d

    fac : int
            factor - new array has multiples of fac as sizes in both dimensions

    sRow : int
            defines # of rows im has to be divisible by

    sCol : int
            defines # of columns im has to be divisible by

    Returns
    -------
    xarray.DataArray
            array with multiples of fac in size, im embedded
    """
    nRows, nColumns = im.shape
    m, n = nRows, nColumns
    # define next bigger size with m,n % fac == 0
    if fac is not None:
        m, n = ((np.array([m, n]) - 1) // int(fac) + 1) * int(fac)
    if sRow is not None:
        m = (m // int(sRow) + 1) * int(m)
    if sCol is not None:
        n = (n // int(sCol) + 1) * int(sCol)
    if m > nRows or n > nColumns:
        im_handle = np.zeros((m, n), dtype=np.float)
        im_handle[:nRows, :nColumns] = im
        return im_handle
    return im


@check_arguments(shape={"image": (0, 0), "xdis": "image", "ydis": "image"})
def map_backward(image, xdis, ydis):
    """
    An image is created by selecting for each position the values of the
    input image at the positions given by the xdis(placement) and ydis(placement).
    The routine works backward - the displacement vectors have to refer to their destination
    - as realized in match_meteosat.pro Hermann Mannstein, May 2003

    Parameters
    ----------
    image : xarray.DataArray or np.ndarray
            image to morph

    xdis : xarray.DataArray or np.ndarray
            displacement in x-direction for each point

    ydis : xarray.DataArray or np.ndarray
            displacement in y-direction for each point

    Returns
    -------
    xarray.DataArray or np.ndarray
            morphed image
    """
    nx = np.shape(image)[1]
    n = np.size(image)

    # seperate displacement in integer (lower value:floor() )  and float values
    # notice: displacement vectors can point in between an area of four surrounding elements
    ixdis = np.array(np.floor(xdis), dtype='int')
    iydis = np.array(np.floor(ydis), dtype='int')
    fxdis = xdis - ixdis
    fydis = ydis - iydis
    fxydis = fxdis * fydis

    # define indexarray for pixel displacement
    ind = np.resize(np.arange(n), np.shape(ixdis))
    # add displacement values to indexarray
    ind = (ind + n + ixdis + iydis * nx) % n  # modulo n for borders of image: wrap around edges

    # now displace pixel image.take(ind)
    # therefor take four surrounding values and weight them with the length of the displacement vectors
    # (weighting factor is float value of displacement vector)
    imdis = (
        (1. - fxdis - fydis + fxydis) * image.take(
            ind) +  # value where the floor value of displacement vector points at
        (fxdis - fxydis) * image.take((ind + n + 1) % n) +  # value right of floor value
        (fydis - fxydis) * image.take((ind + n + nx) % n) +  # value below floor value
        fxydis * image.take((ind + n + nx + 1) % n))  # value right below floor value
    return imdis


def gauss_kern(size, sigma):
    """
    Returns a normalized 2D gauss kernel array for convolutions

    Parameters
    ----------
    size : int
            size of kernel (size=2 refers to a 5x5 kernel)

    sigma : float
            standard deviation of gauss kernel

    Returns
    -------
    array for convolution
    """
    size = int(size)
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x ** 2 / float(size) + y ** 2 / float(size)) / sigma ** 2)
    return g / g.sum()


@check_arguments(shape={"image1": (0, 0), "image2": (0, 0)})
def match_pyramid(image1, image2, factor=4, sigma=5 / 3.):
    """
    Morph image1 that it best matches image2 with pyramidal image matching.

    Parameters
    ----------
    image1 : xarray.DataArray or np.ndarray
            image to morph

    image2 : xarray.DataArray or np.ndarray
            image to match

    factor :  int
            subsampling factor or topmost pyramid level with lowest resolution (averaging 2^factor data points onto one) (default: 4)

    sigma : float
            standard deviation of gauss kernel (default: 5/3.)

    Returns
    -------
    tuple
            morphed image, x-displacement and y-displacement of each data point and least square error of morphed image.
    """
    image1 = np.array(image1, dtype=np.float)
    image2 = np.array(image2, dtype=np.float)
    oS = np.shape(image1)

    # make sure images have powers of 2 as sizes
    im1 = embed_image(image1, 2 ** factor)
    im2 = embed_image(image2, 2 ** factor)
    f1 = im1.copy()

    # initialize displacement vectors
    xdis = np.zeros(np.shape(im1), dtype=float)
    ydis = np.zeros(np.shape(im1), dtype=float)

    ke = gauss_kern(2, sigma)
    zoom = ndimage.zoom  # resize and array with cubic spline interpolation with given order
    convolve = ndimage.convolve  # convolve and array with given weights

    # start pyramide
    for s in range(factor, -1, -1):
        zoomfac = 2. ** s
        n1 = downsize(f1, zoomfac)
        n2 = downsize(im2, zoomfac)

        xd = np.zeros(np.shape(n1), dtype=int)
        yd = np.zeros(np.shape(n1), dtype=int)
        qu = convolve(((n1 - n2) ** 2).astype(np.float), ke, mode='nearest')

        # search in area (-2 to 2) for best dispacement
        for i in range(-2, 3):
            for j in range(-2, 3):
                n1s = np.roll(np.roll(n1, i, axis=1), j, axis=0)  # shift image array, wrap ends around edges
                sm = convolve(((n1s - n2) ** 2).astype(np.float), ke, mode='nearest')
                ind = np.where((sm < qu))
                xd[ind] = i
                yd[ind] = j
                qu[ind] = sm[ind]

        xdis -= zoom(convolve(xd.astype(np.float) * zoomfac, ke, mode='constant'), zoomfac,
                     order=1)  # resize using linear interpolation
        ydis -= zoom(convolve(yd.astype(np.float) * zoomfac, ke, mode='constant'), zoomfac, order=1)

        f1 = map_backward(im1, xdis, ydis)

    lse = convolve(qu, ke, mode='nearest')[:oS[0], :oS[1]]
    #	lse = convolve(((f1-im2)**2).astype(np.float), gauss_kern(2,.5), mode='nearest')[:oS[0],:oS[1]]
    #	lse = ((f1-im2)**2).astype(np.float)[:oS[0],:oS[1]]
    f1 = f1[:oS[0], :oS[1]]
    xdis = -xdis[:oS[0], :oS[1]]
    ydis = -ydis[:oS[0], :oS[1]]

    return f1, xdis, ydis, lse
