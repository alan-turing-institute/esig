# Mechanism for switching backends for computing signatures



import abc
import threading

import numpy

try:
    from esig import tosig
except ImportError:
    # Error occurs during build sequence, since tosig does not exist
    tosig = None


try:
    import iisignature
except ImportError:
    if tosig is None:
        raise ImportError("No available backend for signature calculations, please reinstall esig.")
    iisignature = None


BACKENDS = {}

_BACKEND_LOCK = threading.RLock()
_BACKEND_DEFAULT = None
_BACKEND_CONTAINER = threading.local()


def get_backend():
    """
    Get the current global backend used for computing signatures.
    """
    global _BACKEND_CONTAINER
    if not hasattr(_BACKEND_CONTAINER, "context"):
        with _BACKEND_LOCK:
            _BACKEND_CONTAINER.context = _BACKEND_DEFAULT()
    return _BACKEND_CONTAINER.context


def set_backend(cls_or_name):
    """
    Change the backend used for computing signatures globally.
    """
    if isinstance(cls_or_name, str):
        if not cls_or_name in BACKENDS:
            raise ValueError("%s does not name a valid backend" % cls_or_name)
        _BACKEND_CONTAINER.context = BACKENDS[cls_or_name]()
    elif isinstance(cls_or_name, BackendBase):
        _BACKEND_CONTAINER.context = cls_or_name
    elif issubclass(cls_or_name, BackendBase):
        _BACKEND_CONTAINER.context = cls_or_name()
    else:
        raise TypeError("Backend must be a subclass of the BackendBase class or str")


def set_default_backend(cls):
    """
    Set the default backend across all threads.
    :param cls: Backend class to use.
    """
    with _BACKEND_LOCK:
        _BACKEND_DEFAULT = cls


def list_backends():
    """
    Get a list of all available backends that can be used for computing
    signatures.
    """
    return list(BACKENDS)


class BackendBase(abc.ABC):
    """
    Base class for signature/log signature computation backends.

    The required methods are `compute_signature` and `compute_log_signature`,
    which take data in the form of a Numpy array and return a Numpy array
    containing the flattened signature or log signature.
    """

    @abc.abstractmethod
    def compute_signature(self, stream, depth):
        """
        Compute the signature of the stream to required depth
        """

    @abc.abstractmethod
    def compute_log_signature(self, stream, depth):
        """
        Compute the log signature of the stream to required depth
        """

    def log_sig_dim(self, dimension, depth):
        """
        Get the number of elements in the log signature
        """
        return tosig.logsigdim(dimension, depth)

    def sig_dim(self, dimension, depth):
        """
        Get the number of elements in the signature
        """
        return tosig.sigdim(dimension, depth)

    @abc.abstractmethod
    def log_sig_keys(self, dimension, depth):
        """
        Get the keys that correspond to the elements in the log signature
        """

    @abc.abstractmethod
    def sig_keys(self, dimension, depth):
        """
        Get the keys that correspond to the elements in the signature
        """


class LibalgebraBackend(BackendBase):
    """
    Use libalgebra as a backend for computing signatures and log signatures
    of paths. This is the default option.
    """

    def __repr__(self):
        return "LibalgebraBackend"

    def compute_signature(self, stream, depth):
        return tosig.stream2sig(stream, depth)

    def compute_log_signature(self, stream, depth):
        return tosig.stream2logsig(stream, depth)

    def log_sig_keys(self, dimension, depth):
        return tosig.logsigkeys(dimension, depth)
    
    def sig_keys(self, dimension, depth):
        return tosig.sigkeys(dimension, depth)


BACKENDS["libalgebra"] = LibalgebraBackend


if iisignature:

    class IISignatureBackend(BackendBase):

        def __init__(self):
            self._log_sig_prepare_cache = {}
        
        def __repr__(self):
            return "IISignatureBackend"

        def prepare(self, dimension, depth):
            if (dimension, depth) in self._log_sig_prepare_cache:
                return self._log_sig_prepare_cache[(dimension, depth)]
            
            s = iisignature.prepare(dimension, depth)
            self._log_sig_prepare_cache[(dimension, depth)] = s
            return s

        def compute_signature(self, stream, depth):
            return numpy.concatenate([[1.0], iisignature.sig(stream, depth)], axis=0)

        def compute_log_signature(self, stream, depth):
            _, dim = stream.shape
            s = self.prepare(dim, depth)
            return iisignature.logsig(stream, s)

        def log_sig_keys(self, dimension, depth):
            s = self.prepare(dimension, depth)
            return iisignature.basis(dimension, depth)
        
        def sig_keys(self, dimension, depth):
            return tosig.sigkeys(dimension, depth)




    BACKENDS["iisignature"] = IISignatureBackend


# set the default backend
_BACKEND_DEFAULT = LibalgebraBackend