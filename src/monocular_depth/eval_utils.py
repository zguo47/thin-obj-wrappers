import numpy as np

import torch

def weighted_mean(x, weights=None):
    if weights is None:
        return np.mean(x)
    w = np.clip(weights, 0, None).astype(np.float64)
    if np.sum(w) == 0:
        return np.mean(x)
    return np.sum(x * w) / np.sum(w)


def a1_err(src, tgt, threshold=1.25, weights=None):
    '''
    Accuracy under threshold (e.g., delta < 1.25)

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : percentage of predictions where max(src/tgt, tgt/src) < 1.25
    '''
    eps = 1e-12
    src = np.clip(src, eps, None)
    tgt = np.clip(tgt, eps, None)
    thresh = np.maximum(src / tgt, tgt / src)
    return weighted_mean(thresh < threshold, weights)


def a2_err(src, tgt, threshold=1.25, weights=None):
    '''
    Accuracy under threshold (e.g., delta < 1.25^2)

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : percentage of predictions where max(src/tgt, tgt/src) < 1.25^2
    '''
    eps = 1e-12
    src = np.clip(src, eps, None)
    tgt = np.clip(tgt, eps, None)
    thresh = np.maximum(src / tgt, tgt / src)
    return weighted_mean(thresh < threshold ** 2, weights)


def a3_err(src, tgt, threshold=1.25, weights=None):
    '''
    Accuracy under threshold (e.g., delta < 1.25^3)

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : percentage of predictions where max(src/tgt, tgt/src) < 1.25^3
    '''
    eps = 1e-12
    src = np.clip(src, eps, None)
    tgt = np.clip(tgt, eps, None)
    thresh = np.maximum(src / tgt, tgt / src)
    return weighted_mean(thresh < threshold ** 3, weights)


def log_10_mean_abs_err(src, tgt, weights=None):
    '''
    Mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute error
    '''
    eps = 1e-12
    src = np.clip(src, eps, None)
    tgt = np.clip(tgt, eps, None)
    return weighted_mean(np.abs(np.log10(tgt) - np.log10(src)), weights)


def log_root_mean_sq_err(src, tgt, weights=None):
    '''
    Logarithmic root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : root mean squared error of the log differences
    '''
    eps = 1e-12
    src = np.clip(src, eps, None)
    tgt = np.clip(tgt, eps, None)
    return np.sqrt(weighted_mean((np.log(tgt) - np.log(src)) ** 2, weights))


def root_mean_sq_err(src, tgt, weights=None):
    '''
    Root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : root mean squared error
    '''
    return np.sqrt(weighted_mean((tgt - src) ** 2, weights))


def mean_abs_err(src, tgt, weights=None):
    '''
    Mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute error
    '''
    return weighted_mean(np.abs(tgt - src), weights)


def inv_root_mean_sq_err(src, tgt, weights=None):
    '''
    Inverse root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse root mean squared error
    '''
    return np.sqrt(weighted_mean(((1.0 / tgt) - (1.0 / src)) ** 2, weights))


def inv_mean_abs_err(src, tgt, weights=None):
    '''
    Inverse mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse mean absolute error
    '''
    return weighted_mean(np.abs((1.0 / tgt) - (1.0 / src)), weights)


def abs_rel_err(src, tgt, weights=None):
    '''
    Absolute relative error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : absolute relative error
    '''
    return weighted_mean(np.abs(src - tgt) / tgt, weights)


def sq_rel_err(src, tgt, weights=None):
    '''
    Squared relative error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : squared relative error
    '''
    return weighted_mean(((src - tgt) ** 2) / tgt, weights)


def root_mean_sq_err_torch(src, tgt):
    '''
    Root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : root mean squared error
    '''
    return torch.sqrt(torch.mean((tgt - src) ** 2)).cpu()


def mean_abs_err_torch(src, tgt):
    '''
    Mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute error
    '''
    return torch.mean(torch.abs(tgt - src)).cpu()


def inv_root_mean_sq_err_torch(src, tgt):
    '''
    Inverse root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse root mean squared error
    '''
    return torch.sqrt(torch.mean(((1.0 / tgt) - (1.0 / src)) ** 2)).cpu()


def inv_mean_abs_err_torch(src, tgt):
    '''
    Inverse mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse mean absolute error
    '''
    return torch.mean(torch.abs((1.0 / tgt) - (1.0 / src))).cpu()
