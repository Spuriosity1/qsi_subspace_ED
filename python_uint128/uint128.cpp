#include<Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <string>
#include <stdexcept>


namespace py = pybind11;


// Utility: extract __uint128_t from Python int
static __uint128_t extract_uint128(py::handle obj) {
    if (!PyLong_Check(obj.ptr())) {
        throw std::invalid_argument("Expected Python int");
    } PyObject *pyint = obj.ptr();

    // Extract lower and upper 64 bits
    uint64_t low = PyLong_AsUnsignedLongLongMask(pyint);
    PyObject *shifted = PyNumber_Rshift(pyint, PyLong_FromLong(64));
    uint64_t high = PyLong_AsUnsignedLongLongMask(shifted);
    Py_DECREF(shifted);

    __uint128_t result = (__uint128_t(high) << 64) | low;
    return result;
}







// Helper function to convert __int128 to string
inline std::string uint128_to_string(__uint128_t value) {
    std::string result =  "0x00000000000000000000000000000000";
    const char* hex_chars= "0123456789abcdef";
    unsigned pos=0;
	while (value > 0) {
        result[33 - pos] = hex_chars[value%16];
		value >>= 4;
        pos++;
	}
	return result;
}


class UInt128 {
public:
	__uint128_t value;

	UInt128() : value(0) {}

	explicit UInt128(uint64_t low) : value(static_cast<__uint128_t>(low)) {}

	explicit UInt128(__uint128_t val) : value(val) {}

	explicit UInt128(std::string str); 

	std::string str() const {
		return uint128_to_string(value);
	}

	uint64_t low64() const {
		return static_cast<uint64_t>(value);
	}

	uint64_t high64() const {
		return static_cast<uint64_t>(value >> 64);
	}

	// Operators
	UInt128 operator+(const UInt128& other) const {
		return UInt128(this->value + other.value);
	}

	UInt128 operator-(const UInt128& other) const {
		return UInt128(this->value - other.value);
	}

	UInt128 operator&(const UInt128& other) const {
		return UInt128(this->value & other.value);
	}

	UInt128 operator|(const UInt128& other) const {
		return UInt128(this->value | other.value);
	}

	UInt128 operator^(const UInt128& other) const {
		return UInt128(this->value ^ other.value);
	}


	UInt128 operator|=(const UInt128& other) {
		this->value |= other.value;
        return *this;
	}


	UInt128 operator&=(const UInt128& other) {
		this->value &= other.value;
        return *this;
	}


	UInt128 operator^=(const UInt128& other) {
		this->value ^= other.value;
        return *this;
	}
	
	template<typename T>
	UInt128 operator<<(T other) const {
		return UInt128(this->value << other);
	}

	template<typename T>
	UInt128 operator>>(T other) const {
		return UInt128(this->value >> other);
	}

	bool operator==(const UInt128& other) const {
		return this->value == other.value;
	}

	bool operator<(const UInt128& other) const {
		return this->value < other.value;
	}

	bool operator>(const UInt128& other) const {
		return this->value > other.value;
	}

};


UInt128::UInt128(std::string str) {
    value = 0;

    // Check if the string starts with "0x" for hexadecimal
    bool is_hex = (str.size() > 2 && str[0] == '0' && (str[1] == 'x' || str[1] == 'X'));

    // Remove "0x" prefix for hex strings
    if (is_hex) {
        str = str.substr(2);
    }

    // Iterate over the string and process the characters
    for (char c : str) {
        int digit_value;

        // Check if it's a valid hexadecimal or decimal character
        if (c >= '0' && c <= '9') {
            digit_value = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            digit_value = c - 'a' + 10;
        } else if (c >= 'A' && c <= 'F') {
            digit_value = c - 'A' + 10;
        } else {
            throw std::invalid_argument("Invalid character in UInt128 string");
        }

        // Update the value, multiplying by the base (10 or 16) each time
        value = value * (is_hex ? 16 : 10) + digit_value;
    }
}

// Hash function for __uint128_t (e.g., split into 64-bit halves)
std::size_t uint128_hash(const __uint128_t& val) {
    uint64_t high = static_cast<uint64_t>(val >> 64);
    uint64_t low  = static_cast<uint64_t>(val);
    // Combine the two halves using std::hash
    return std::hash<uint64_t>()(low) ^ (std::hash<uint64_t>()(high) << 1);
}



PYBIND11_MODULE(uint128, m) {
  py::class_<UInt128>(m, "UInt128")
      .def(py::init<>())
      .def(py::init<uint64_t>())
      .def(py::init([](py::int_ value) {
            return UInt128(extract_uint128(value));
    }), "Create from Python int of arbitrary size")
      .def(py::init<std::string>(), "Create from decimal string")
      .def("__str__", &UInt128::str)
      .def("__repr__",
           [](const UInt128 &v) { return "<UInt128 " + v.str() + ">"; })
      .def("__int__",
           [](const UInt128 &v) {
             return py::reinterpret_steal<py::object>(
                 PyLong_FromString(v.str().c_str(), nullptr, 16));
           })
      .def("low64", &UInt128::low64)
      .def("high64", &UInt128::high64)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self == py::self)
      .def("__hash__", [](const UInt128& self){
            return uint128_hash(self.value);
        })
      .def(py::self < py::self)
      .def(py::self > py::self)
      .def(py::self | py::self)
      .def(py::self |= py::self)
      .def(py::self ^ py::self)
      .def(py::self ^= py::self)
      .def(py::self & py::self)
      .def(py::self &= py::self)
      // shift by Python int
      .def("__lshift__", [](const UInt128 &a, py::int_ shift) {
        unsigned long long s = shift.cast<unsigned long long>();
        if (s > 127)
          throw std::invalid_argument("Shift exceeds 127 bits");
        return a << static_cast<unsigned int>(s);
      })
      .def("__rshift__", [](const UInt128 &a, py::int_ shift) {
        unsigned long long s = shift.cast<unsigned long long>();
        if (s > 127)
          throw std::invalid_argument("Shift exceeds 127 bits");
        return a >> static_cast<unsigned int>(s);
      });
}

