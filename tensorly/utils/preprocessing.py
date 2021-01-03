from .. import backend as T
from math import sqrt


def _is_iterable(x):
    try:
        iter(x)
    except TypeError:
        return False
    return True


def std(tensor, axis=None, keepdims=False, ddof=0):
    """Compute the standard deviation along the specified axes.

    Returns the standard deviation, a measure of the spread of a distribution,
    of the array elements. The standard deviation is computed for the flattened
    array by default, otherwise over the specified axes.
    """
    shape = T.shape(tensor)
    if axis is None:
        axis = list(range(len(shape)))
    elif not _is_iterable(axis):
        axis = [axis]
    
    reduce_axes = (i for i in range(len(shape)) if i in axis)
    dofs = 1
    for axis in reduce_axes:
        dofs *= shape[axis]
    dofs -= ddof

    return T.norm(tensor - T.mean) / sqrt(dofs)


class ProperCenterAndScale:
    """Proper centering of the tensor :math:`X` across the specified modes.

    In [1]_, Rasmus Bro and Age K. Smilde define proper centering and scaling
    of a dataset. Proper centering will, in some cases, not affect the fitted
    components and proper scaling is equivalent to fitting a model with a weighted
    loss function. An important note is that we should not center across the
    mode we scale within, since then, the scaling will affect the centering.
    
    Centering
    ^^^^^^^^^

    Centering is performed to remove constant offsets in the measurements and
    should be performed "across" a mode. The centering of a third order tensor
    :math:`X` across the third mode is defined as

    .. code::python

        X_center = X - mean(X, axis=2, keepdims=True)

    Likeways, the cenering of :math:`X` across the first two modes is defined as

    .. code::python

        tmp = X - mean(X, axis=0, keepdims=True)
        X_center = tmp - mean(tmp, axis=1, keepdims=True)

    Note that if the dataset 

    Scaling
    ^^^^^^^

    Scaling is normally performed to ensure that the sub tensors within a mode
    all have unit variance or unit norm. The operation should be performed "within"
    modes, and preferably only within a single mode. This is beause we may not
    be able to find a tensor with unit variance (or norm) within multiple modes
    only by rescaling a tensor.
    
    Scaling of a third order tensor :math:`X` within the first mode is defined as

    .. code::python

        X_scale = X/std(X, axis=(1, 2), keepdims=True)

    Likeways, the scaling of :math:`X` within the first two modes is defined as

    .. code::python

        tmp = X/std(X, axis=(1, 2), keepdims=True)
        X_scale = tmp/std(tmp, axis=(0, 2), keepdims=True)
    
    However, the above expression is not guaranteed to be equal to

    .. code::python

        tmp = X/std(X, axis=(0, 2), keepdims=True)
        X_scale = tmp/std(tmp, axis=(1, 2), keepdims=True)
    
    which means that scaling within multiple modes are dependent on the order
    of the scaling.

    Arguments
    ---------
    center_across : int or iterable of ints
        The mode(s) to center across, should not overlap with any mode supplied
        to ``scale_within``
    scale_within : int or iterable of ints
        The mode(s) to scale within, should not overlap with any mode supplied
        to ``center_across``
    scale_weight : str or callable [default="std"]
        The weight function to use for scaling. Can be one of the strings
        ``"std"`` and ``"norm"``, or a callable whose first argument is a tensor,
        and accepts the keyword arguments ``axis`` and ``keepdims``.
    allow_problematic_scaling : bool [default=False]
        Disable check for problematic scaling, if `False` then a ValueError
        will be raised
    **weight_kwargs : 
        Additional keyword arguments passed to the scale_weight function.
        For example ``ddof=1`` with ``scale_weight="std"`` will give an
        unbiased estimator of the distribution standard deviation instead
        of the sample standard deviation (which is default).

    References
    ----------
    .. [1] Bro, R. and Smilde, A.K. (2003),
            Centering and scaling in component analysis.
            J. Chemometrics, 17: 16-33.
    """
    def __init__(self, center_across, scale_within, scale_weight="std",
                 allow_problematic_scaling=False, **weight_kwargs):
        self.center_across = center_across
        self.scale_within = scale_within
        self.scale_weight = scale_weight
        self.allow_problematic_scaling = allow_problematic_scaling
        self.weight_kwargs = weight_kwargs
