# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

import os; saved_cwd = os.getcwd();	os.chdir(os.path.dirname(os.path.realpath(__file__))) 


from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _letsberational
else:
    import _letsberational

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

_swig_new_instance_method = _letsberational.SWIG_PyInstanceMethod_New
_swig_new_static_method = _letsberational.SWIG_PyStaticMethod_New

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


Black = _letsberational.Black
NormalisedBlack = _letsberational.NormalisedBlack
ImpliedBlackVolatility = _letsberational.ImpliedBlackVolatility
NormalisedImpliedBlackVolatility = _letsberational.NormalisedImpliedBlackVolatility
ComplementaryNormalisedBlack = _letsberational.ComplementaryNormalisedBlack
Vega = _letsberational.Vega
NormalisedVega = _letsberational.NormalisedVega
Volga = _letsberational.Volga
NormalisedVolga = _letsberational.NormalisedVolga
BlackAccuracyFactor = _letsberational.BlackAccuracyFactor
ImpliedVolatilityAttainableAccuracy = _letsberational.ImpliedVolatilityAttainableAccuracy
DblEpsilon = _letsberational.DblEpsilon
DblMin = _letsberational.DblMin
DblMax = _letsberational.DblMax
CPUName = _letsberational.CPUName
Revision = _letsberational.Revision
os.chdir(saved_cwd); saved_cwd 


