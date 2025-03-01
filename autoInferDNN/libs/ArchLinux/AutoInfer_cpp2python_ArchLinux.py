# This file was automatically generated by SWIG (https://www.swig.org).
# Version 4.1.1
#
# Do not make changes to this file unless you know what you are doing - modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _AutoInfer_cpp2python_ArchLinux
else:
    import _AutoInfer_cpp2python_ArchLinux

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "this":
            set(self, name, value)
        elif name == "thisown":
            self.this.own(value)
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


class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _AutoInfer_cpp2python_ArchLinux.delete_SwigPyIterator

    def value(self):
        return _AutoInfer_cpp2python_ArchLinux.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _AutoInfer_cpp2python_ArchLinux.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _AutoInfer_cpp2python_ArchLinux.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _AutoInfer_cpp2python_ArchLinux.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _AutoInfer_cpp2python_ArchLinux.SwigPyIterator_equal(self, x)

    def copy(self):
        return _AutoInfer_cpp2python_ArchLinux.SwigPyIterator_copy(self)

    def next(self):
        return _AutoInfer_cpp2python_ArchLinux.SwigPyIterator_next(self)

    def __next__(self):
        return _AutoInfer_cpp2python_ArchLinux.SwigPyIterator___next__(self)

    def previous(self):
        return _AutoInfer_cpp2python_ArchLinux.SwigPyIterator_previous(self)

    def advance(self, n):
        return _AutoInfer_cpp2python_ArchLinux.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _AutoInfer_cpp2python_ArchLinux.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _AutoInfer_cpp2python_ArchLinux.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _AutoInfer_cpp2python_ArchLinux.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _AutoInfer_cpp2python_ArchLinux.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _AutoInfer_cpp2python_ArchLinux.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _AutoInfer_cpp2python_ArchLinux:
_AutoInfer_cpp2python_ArchLinux.SwigPyIterator_swigregister(SwigPyIterator)
class IntVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _AutoInfer_cpp2python_ArchLinux.IntVector___nonzero__(self)

    def __bool__(self):
        return _AutoInfer_cpp2python_ArchLinux.IntVector___bool__(self)

    def __len__(self):
        return _AutoInfer_cpp2python_ArchLinux.IntVector___len__(self)

    def __getslice__(self, i, j):
        return _AutoInfer_cpp2python_ArchLinux.IntVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.IntVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _AutoInfer_cpp2python_ArchLinux.IntVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.IntVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.IntVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.IntVector___setitem__(self, *args)

    def pop(self):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_pop(self)

    def append(self, x):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_append(self, x)

    def empty(self):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_empty(self)

    def size(self):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_size(self)

    def swap(self, v):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_swap(self, v)

    def begin(self):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_begin(self)

    def end(self):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_end(self)

    def rbegin(self):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_rbegin(self)

    def rend(self):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_rend(self)

    def clear(self):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_clear(self)

    def get_allocator(self):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_get_allocator(self)

    def pop_back(self):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_pop_back(self)

    def erase(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_erase(self, *args)

    def __init__(self, *args):
        _AutoInfer_cpp2python_ArchLinux.IntVector_swiginit(self, _AutoInfer_cpp2python_ArchLinux.new_IntVector(*args))

    def push_back(self, x):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_push_back(self, x)

    def front(self):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_front(self)

    def back(self):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_back(self)

    def assign(self, n, x):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_assign(self, n, x)

    def resize(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_resize(self, *args)

    def insert(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_insert(self, *args)

    def reserve(self, n):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_reserve(self, n)

    def capacity(self):
        return _AutoInfer_cpp2python_ArchLinux.IntVector_capacity(self)
    __swig_destroy__ = _AutoInfer_cpp2python_ArchLinux.delete_IntVector

# Register IntVector in _AutoInfer_cpp2python_ArchLinux:
_AutoInfer_cpp2python_ArchLinux.IntVector_swigregister(IntVector)
class UInt32Vector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector___nonzero__(self)

    def __bool__(self):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector___bool__(self)

    def __len__(self):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector___len__(self)

    def __getslice__(self, i, j):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector___setitem__(self, *args)

    def pop(self):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_pop(self)

    def append(self, x):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_append(self, x)

    def empty(self):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_empty(self)

    def size(self):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_size(self)

    def swap(self, v):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_swap(self, v)

    def begin(self):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_begin(self)

    def end(self):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_end(self)

    def rbegin(self):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_rbegin(self)

    def rend(self):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_rend(self)

    def clear(self):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_clear(self)

    def get_allocator(self):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_get_allocator(self)

    def pop_back(self):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_pop_back(self)

    def erase(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_erase(self, *args)

    def __init__(self, *args):
        _AutoInfer_cpp2python_ArchLinux.UInt32Vector_swiginit(self, _AutoInfer_cpp2python_ArchLinux.new_UInt32Vector(*args))

    def push_back(self, x):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_push_back(self, x)

    def front(self):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_front(self)

    def back(self):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_back(self)

    def assign(self, n, x):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_assign(self, n, x)

    def resize(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_resize(self, *args)

    def insert(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_insert(self, *args)

    def reserve(self, n):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_reserve(self, n)

    def capacity(self):
        return _AutoInfer_cpp2python_ArchLinux.UInt32Vector_capacity(self)
    __swig_destroy__ = _AutoInfer_cpp2python_ArchLinux.delete_UInt32Vector

# Register UInt32Vector in _AutoInfer_cpp2python_ArchLinux:
_AutoInfer_cpp2python_ArchLinux.UInt32Vector_swigregister(UInt32Vector)
class StringVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _AutoInfer_cpp2python_ArchLinux.StringVector___nonzero__(self)

    def __bool__(self):
        return _AutoInfer_cpp2python_ArchLinux.StringVector___bool__(self)

    def __len__(self):
        return _AutoInfer_cpp2python_ArchLinux.StringVector___len__(self)

    def __getslice__(self, i, j):
        return _AutoInfer_cpp2python_ArchLinux.StringVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.StringVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _AutoInfer_cpp2python_ArchLinux.StringVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.StringVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.StringVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.StringVector___setitem__(self, *args)

    def pop(self):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_pop(self)

    def append(self, x):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_append(self, x)

    def empty(self):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_empty(self)

    def size(self):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_size(self)

    def swap(self, v):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_swap(self, v)

    def begin(self):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_begin(self)

    def end(self):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_end(self)

    def rbegin(self):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_rbegin(self)

    def rend(self):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_rend(self)

    def clear(self):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_clear(self)

    def get_allocator(self):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_get_allocator(self)

    def pop_back(self):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_pop_back(self)

    def erase(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_erase(self, *args)

    def __init__(self, *args):
        _AutoInfer_cpp2python_ArchLinux.StringVector_swiginit(self, _AutoInfer_cpp2python_ArchLinux.new_StringVector(*args))

    def push_back(self, x):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_push_back(self, x)

    def front(self):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_front(self)

    def back(self):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_back(self)

    def assign(self, n, x):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_assign(self, n, x)

    def resize(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_resize(self, *args)

    def insert(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_insert(self, *args)

    def reserve(self, n):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_reserve(self, n)

    def capacity(self):
        return _AutoInfer_cpp2python_ArchLinux.StringVector_capacity(self)
    __swig_destroy__ = _AutoInfer_cpp2python_ArchLinux.delete_StringVector

# Register StringVector in _AutoInfer_cpp2python_ArchLinux:
_AutoInfer_cpp2python_ArchLinux.StringVector_swigregister(StringVector)
class FloatVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector___nonzero__(self)

    def __bool__(self):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector___bool__(self)

    def __len__(self):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector___len__(self)

    def __getslice__(self, i, j):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector___setitem__(self, *args)

    def pop(self):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_pop(self)

    def append(self, x):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_append(self, x)

    def empty(self):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_empty(self)

    def size(self):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_size(self)

    def swap(self, v):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_swap(self, v)

    def begin(self):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_begin(self)

    def end(self):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_end(self)

    def rbegin(self):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_rbegin(self)

    def rend(self):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_rend(self)

    def clear(self):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_clear(self)

    def get_allocator(self):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_get_allocator(self)

    def pop_back(self):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_pop_back(self)

    def erase(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_erase(self, *args)

    def __init__(self, *args):
        _AutoInfer_cpp2python_ArchLinux.FloatVector_swiginit(self, _AutoInfer_cpp2python_ArchLinux.new_FloatVector(*args))

    def push_back(self, x):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_push_back(self, x)

    def front(self):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_front(self)

    def back(self):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_back(self)

    def assign(self, n, x):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_assign(self, n, x)

    def resize(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_resize(self, *args)

    def insert(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_insert(self, *args)

    def reserve(self, n):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_reserve(self, n)

    def capacity(self):
        return _AutoInfer_cpp2python_ArchLinux.FloatVector_capacity(self)
    __swig_destroy__ = _AutoInfer_cpp2python_ArchLinux.delete_FloatVector

# Register FloatVector in _AutoInfer_cpp2python_ArchLinux:
_AutoInfer_cpp2python_ArchLinux.FloatVector_swigregister(FloatVector)
class DoubleVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector___nonzero__(self)

    def __bool__(self):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector___bool__(self)

    def __len__(self):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector___len__(self)

    def __getslice__(self, i, j):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector___setitem__(self, *args)

    def pop(self):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_pop(self)

    def append(self, x):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_append(self, x)

    def empty(self):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_empty(self)

    def size(self):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_size(self)

    def swap(self, v):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_swap(self, v)

    def begin(self):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_begin(self)

    def end(self):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_end(self)

    def rbegin(self):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_rbegin(self)

    def rend(self):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_rend(self)

    def clear(self):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_clear(self)

    def get_allocator(self):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_get_allocator(self)

    def pop_back(self):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_pop_back(self)

    def erase(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_erase(self, *args)

    def __init__(self, *args):
        _AutoInfer_cpp2python_ArchLinux.DoubleVector_swiginit(self, _AutoInfer_cpp2python_ArchLinux.new_DoubleVector(*args))

    def push_back(self, x):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_push_back(self, x)

    def front(self):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_front(self)

    def back(self):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_back(self)

    def assign(self, n, x):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_assign(self, n, x)

    def resize(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_resize(self, *args)

    def insert(self, *args):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_insert(self, *args)

    def reserve(self, n):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_reserve(self, n)

    def capacity(self):
        return _AutoInfer_cpp2python_ArchLinux.DoubleVector_capacity(self)
    __swig_destroy__ = _AutoInfer_cpp2python_ArchLinux.delete_DoubleVector

# Register DoubleVector in _AutoInfer_cpp2python_ArchLinux:
_AutoInfer_cpp2python_ArchLinux.DoubleVector_swigregister(DoubleVector)
input_i1 = _AutoInfer_cpp2python_ArchLinux.input_i1
input_i2 = _AutoInfer_cpp2python_ArchLinux.input_i2
output_o1 = _AutoInfer_cpp2python_ArchLinux.output_o1
output_o2 = _AutoInfer_cpp2python_ArchLinux.output_o2
PRECISION = _AutoInfer_cpp2python_ArchLinux.PRECISION
MAGIC_EXCLUDE_double = _AutoInfer_cpp2python_ArchLinux.MAGIC_EXCLUDE_double

def actvFunSigmoid(_in):
    return _AutoInfer_cpp2python_ArchLinux.actvFunSigmoid(_in)

def actvFunTanH(_in):
    return _AutoInfer_cpp2python_ArchLinux.actvFunTanH(_in)

def derivativeCalc(funPtr, fun_in):
    return _AutoInfer_cpp2python_ArchLinux.derivativeCalc(funPtr, fun_in)

def getActivationFun(funName):
    return _AutoInfer_cpp2python_ArchLinux.getActivationFun(funName)
class Eason(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _AutoInfer_cpp2python_ArchLinux.Eason_swiginit(self, _AutoInfer_cpp2python_ArchLinux.new_Eason())
    __swig_destroy__ = _AutoInfer_cpp2python_ArchLinux.delete_Eason
    input = property(_AutoInfer_cpp2python_ArchLinux.Eason_input_get, _AutoInfer_cpp2python_ArchLinux.Eason_input_set)
    target_output = property(_AutoInfer_cpp2python_ArchLinux.Eason_target_output_get, _AutoInfer_cpp2python_ArchLinux.Eason_target_output_set)

    def setWeightBias(self, weights_h1n2_in, bias_h1n2_in, weights_end_in, bias_end_in, learning_rate_in):
        return _AutoInfer_cpp2python_ArchLinux.Eason_setWeightBias(self, weights_h1n2_in, bias_h1n2_in, weights_end_in, bias_end_in, learning_rate_in)

    def setInput_TargetOutput(self, input, target_output):
        return _AutoInfer_cpp2python_ArchLinux.Eason_setInput_TargetOutput(self, input, target_output)

    def train(self, turns=20, logOut=True, autoTrain=False, mse_threshold=0.0):
        return _AutoInfer_cpp2python_ArchLinux.Eason_train(self, turns, logOut, autoTrain, mse_threshold)

# Register Eason in _AutoInfer_cpp2python_ArchLinux:
_AutoInfer_cpp2python_ArchLinux.Eason_swigregister(Eason)

