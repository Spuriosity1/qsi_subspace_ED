# distutils: language = c++
cdef extern from *:
    """
    typedef __int128 int128_t;
    typedef unsigned __int128 uint128_t;
    """
    ctypedef long int128_t
    ctypedef unsigned long uint128_t

cdef class UInt128:
    cdef uint128_t value

    def __cinit__(self, val):
        if isinstance(val, int):
            self.value = <uint128_t>val
        else:
            raise TypeError("UInt128 expects an int")

    def __int__(self):
        return int(self.value)

    def __repr__(self):
        return f"UInt128({int(self.value)})"

    def __add__(self, other):
        if isinstance(other, UInt128):
            return UInt128(int(self.value + other.value))
        elif isinstance(other, int):
            return UInt128(int(self.value + other))
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, UInt128):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        return NotImplemented
