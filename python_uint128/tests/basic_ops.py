import sys
import os


sys.path.append(os.path.abspath("build/uint128_pybind/"))

print(sys.path)

from uint128 import UInt128

B = UInt128(1)

A = UInt128(1 << 65 | 1)

C = UInt128("183927380171882187")

print("A= ",A)
print("B= ",B)
print("C= ",C)

print(f"int(A) = {int(A)}")

print("A+B ", A+B)
print("A-B ", A-B)
print("A==B ", A==B)
print("A<B ", A<B)
print("A>B ", A>B)
print("A|B ", A|B)
print("A&B ", A&B)
print("A^B ", A^B)


print("A<<3 ", A<<3)

print("A>>3 ", A>>3)
