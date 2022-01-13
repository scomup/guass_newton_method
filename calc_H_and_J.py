import sympy
sympy.init_printing()
tx = sympy.Symbol('tx')
ty = sympy.Symbol('ty')
tz = sympy.Symbol('tz')
rx = sympy.Symbol('rx')
ry = sympy.Symbol('ry')
rz = sympy.Symbol('rz')

x1 = sympy.Symbol('x1')
x2 = sympy.Symbol('x2')
x3 = sympy.Symbol('x3')

T00 = sympy.Symbol('T00')
T01 = sympy.Symbol('T01')
T02 = sympy.Symbol('T02')
T03 = sympy.Symbol('T03')
T10 = sympy.Symbol('T10')
T11 = sympy.Symbol('T11')
T12 = sympy.Symbol('T12')
T13 = sympy.Symbol('T13')
T20 = sympy.Symbol('T20')
T21 = sympy.Symbol('T21')
T22 = sympy.Symbol('T22')
T23 = sympy.Symbol('T23')


cx = sympy.cos(rx)
sx = sympy.sin(rx)
cy = sympy.cos(ry)
sy = sympy.sin(ry)
cz = sympy.cos(rz)
sz = sympy.sin(rz)

x_prime0 = (cy*cz) * x1 + (-cy*sz) *x2 + (sy)*x3 + tx
y_prime0 = (cx*sz+sx*sy*cz) * x1 + (cx*cz-sx*sy*sz) *x2 + (-sx*cy)*x3 + ty
z_prime0 = (sx*sz-cx*sy*cz) * x1 + (cx*sy*sz+sx*cz) *x2 + (cx*cy)*x3 + tz
x_prime1 = T00*x_prime0 + T01*y_prime0 + T02*z_prime0 + T03
y_prime1 = T10*x_prime0 + T11*y_prime0 + T12*z_prime0 + T13
z_prime1 = T20*x_prime0 + T21*y_prime0 + T22*z_prime0 + T23

J =  sympy.Matrix([[sympy.diff(x_prime1, tx),sympy.diff(x_prime1, ty),sympy.diff(x_prime1, tz),sympy.diff(x_prime1, rx),sympy.diff(x_prime1, ry),sympy.diff(x_prime1, rz)],
                   [sympy.diff(y_prime1, tx),sympy.diff(y_prime1, ty),sympy.diff(y_prime1, tz),sympy.diff(y_prime1, rx),sympy.diff(y_prime1, ry),sympy.diff(y_prime1, rz)],
                   [sympy.diff(z_prime1, tx),sympy.diff(z_prime1, ty),sympy.diff(z_prime1, tz),sympy.diff(z_prime1, rx),sympy.diff(z_prime1, ry),sympy.diff(z_prime1, rz)],
])

H =  sympy.Matrix([[sympy.diff(J[0,0], tx),sympy.diff(J[0,1], tx),sympy.diff(J[0,2], tx),sympy.diff(J[0,3], tx),sympy.diff(J[0,4], tx),sympy.diff(J[0,5], tx)],
                   [sympy.diff(J[1,0], tx),sympy.diff(J[1,1], tx),sympy.diff(J[1,2], tx),sympy.diff(J[1,3], tx),sympy.diff(J[1,4], tx),sympy.diff(J[1,5], tx)],
                   [sympy.diff(J[2,0], tx),sympy.diff(J[2,1], tx),sympy.diff(J[2,2], tx),sympy.diff(J[2,3], tx),sympy.diff(J[2,4], tx),sympy.diff(J[2,5], tx)],
                   [sympy.diff(J[0,0], ty),sympy.diff(J[0,1], ty),sympy.diff(J[0,2], ty),sympy.diff(J[0,3], ty),sympy.diff(J[0,4], ty),sympy.diff(J[0,5], ty)],
                   [sympy.diff(J[1,0], ty),sympy.diff(J[1,1], ty),sympy.diff(J[1,2], ty),sympy.diff(J[1,3], ty),sympy.diff(J[1,4], ty),sympy.diff(J[1,5], ty)],
                   [sympy.diff(J[2,0], ty),sympy.diff(J[2,1], ty),sympy.diff(J[2,2], ty),sympy.diff(J[2,3], ty),sympy.diff(J[2,4], ty),sympy.diff(J[2,5], ty)],
                   [sympy.diff(J[0,0], tz),sympy.diff(J[0,1], tz),sympy.diff(J[0,2], tz),sympy.diff(J[0,3], tz),sympy.diff(J[0,4], tz),sympy.diff(J[0,5], tz)],
                   [sympy.diff(J[1,0], tz),sympy.diff(J[1,1], tz),sympy.diff(J[1,2], tz),sympy.diff(J[1,3], tz),sympy.diff(J[1,4], tz),sympy.diff(J[1,5], tz)],
                   [sympy.diff(J[2,0], tz),sympy.diff(J[2,1], tz),sympy.diff(J[2,2], tz),sympy.diff(J[2,3], tz),sympy.diff(J[2,4], tz),sympy.diff(J[2,5], tz)],
                   [sympy.diff(J[0,0], rx),sympy.diff(J[0,1], rx),sympy.diff(J[0,2], rx),sympy.diff(J[0,3], rx),sympy.diff(J[0,4], rx),sympy.diff(J[0,5], rx)],
                   [sympy.diff(J[1,0], rx),sympy.diff(J[1,1], rx),sympy.diff(J[1,2], rx),sympy.diff(J[1,3], rx),sympy.diff(J[1,4], rx),sympy.diff(J[1,5], rx)],
                   [sympy.diff(J[2,0], rx),sympy.diff(J[2,1], rx),sympy.diff(J[2,2], rx),sympy.diff(J[2,3], rx),sympy.diff(J[2,4], rx),sympy.diff(J[2,5], rx)],[sympy.diff(J[0,0], tx),sympy.diff(J[0,1], tx),sympy.diff(J[0,2], tx),sympy.diff(J[0,3], tx),sympy.diff(J[0,4], tx),sympy.diff(J[0,5], tx)],
                   [sympy.diff(J[1,0], rx),sympy.diff(J[1,1], rx),sympy.diff(J[1,2], rx),sympy.diff(J[1,3], rx),sympy.diff(J[1,4], rx),sympy.diff(J[1,5], rx)],
                   [sympy.diff(J[2,0], rx),sympy.diff(J[2,1], rx),sympy.diff(J[2,2], rx),sympy.diff(J[2,3], rx),sympy.diff(J[2,4], rx),sympy.diff(J[2,5], rx)],[sympy.diff(J[0,0], tx),sympy.diff(J[0,1], tx),sympy.diff(J[0,2], tx),sympy.diff(J[0,3], tx),sympy.diff(J[0,4], tx),sympy.diff(J[0,5], tx)],
                   [sympy.diff(J[1,0], rx),sympy.diff(J[1,1], rx),sympy.diff(J[1,2], rx),sympy.diff(J[1,3], rx),sympy.diff(J[1,4], rx),sympy.diff(J[1,5], rx)],
                   [sympy.diff(J[2,0], rx),sympy.diff(J[2,1], rx),sympy.diff(J[2,2], rx),sympy.diff(J[2,3], rx),sympy.diff(J[2,4], rx),sympy.diff(J[2,5], rx)],
])

print(J.subs([(rx,0),(ry,0),(rz,0)]))
print(H.subs([(rx,0),(ry,0),(rz,0)]))
