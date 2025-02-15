import numpy as np

def Jring(Jpm, B):

    Bx, By, Bz = B
    return np.array( [1/36 * ( -21/4 * ( ( Bx + ( By + -1 * Bz ) ) )**2 * ( ( \
Bx + ( -1 * By + Bz ) ) )**2 * ( ( -1 * Bx + ( By + Bz ) ) )**2 + ( \
-35/2 * ( Bx + ( -1 * By + -1 * Bz ) ) * ( Bx + ( By + -1 * Bz ) ) * \
( Bx + ( -1 * By + Bz ) ) * ( Bx + ( By + Bz ) ) * Jpm + ( -15 * ( ( \
Bx + ( By + Bz ) ) )**2 * Jpm**2 + 54 * Jpm**3 ) ) ),1/36 * ( -21/4 * \
( ( Bx + ( By + -1 * Bz ) ) )**2 * ( ( Bx + ( -1 * By + Bz ) ) )**2 * \
( ( Bx + ( By + Bz ) ) )**2 + ( -35/2 * ( Bx + ( -1 * By + -1 * Bz ) \
) * ( Bx + ( By + -1 * Bz ) ) * ( Bx + ( -1 * By + Bz ) ) * ( Bx + ( \
By + Bz ) ) * Jpm + ( -15 * ( ( -1 * Bx + ( By + Bz ) ) )**2 * Jpm**2 \
+ 54 * Jpm**3 ) ) ),1/36 * ( -21/4 * ( ( Bx + ( By + -1 * Bz ) ) )**2 \
* ( ( -1 * Bx + ( By + Bz ) ) )**2 * ( ( Bx + ( By + Bz ) ) )**2 + ( \
-35/2 * ( Bx + ( -1 * By + -1 * Bz ) ) * ( Bx + ( By + -1 * Bz ) ) * \
( Bx + ( -1 * By + Bz ) ) * ( Bx + ( By + Bz ) ) * Jpm + ( -15 * ( ( \
Bx + ( -1 * By + Bz ) ) )**2 * Jpm**2 + 54 * Jpm**3 ) ) ),1/36 * ( \
-21/4 * ( ( Bx + ( -1 * By + Bz ) ) )**2 * ( ( -1 * Bx + ( By + Bz ) \
) )**2 * ( ( Bx + ( By + Bz ) ) )**2 + ( -35/2 * ( Bx + ( -1 * By + \
-1 * Bz ) ) * ( Bx + ( By + -1 * Bz ) ) * ( Bx + ( -1 * By + Bz ) ) * \
( Bx + ( By + Bz ) ) * Jpm + ( -15 * ( ( Bx + ( By + -1 * Bz ) ) )**2 \
* Jpm**2 + 54 * Jpm**3 ) ) ),] )