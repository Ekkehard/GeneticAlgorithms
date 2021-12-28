import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
try:
    __IPYTHON__
except NameError:
    iPython = False
else:
    iPython = True
    sy.init_printing()

def objf( x, coeffs ):
    if type( x ) is list:
        for xx in x:
            power = 0
            retval = 0
            for coeff in coeffs[::-1]:
                retval += coeff * xx**power
                power += 1
    else:
        power = 0
        retval = 0
        for coeff in coeffs[::-1]:
            retval += coeff * x**power
            power += 1
    return retval

# fixed parameters
x0 = 0.0
y0 = 0.0
y1 = 1.0
x4 = 1.0
y4 = 0.0

# adjustable parameters
x1 = 0.15
x2 = 0.43
x3 = 0.7

a, b, c, d, e, f = sy.symbols( "a, b, c, d, e, f", real=True )

x, y, y_prime, y_2prime, y_3prime = \
    sy.symbols( "x, y, y', y'', y'''", real=True )


y = a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f
y_prime = sy.diff( y, x )
y_2prime = sy.diff( y_prime, x )
y_3prime = sy.diff( y_2prime, x )

print( "y = {0}".format( y ) )
print( "y' = {0}".format( y_prime ) )
print( "y'' = {0}".format( y_2prime ) )
print( "y''' = {0}".format( y_3prime ) )

Eqns=[y.subs({x:x0})-y0,
      y.subs({x:x1})-y1,
      y.subs({x:x4})-y4,
      y_prime.subs({x:x1}),
      y_prime.subs({x:x2}),
      y_prime.subs({x:x3})]

res = sy.linsolve( Eqns, (a, b, c, d, e, f) )

if len( res ) > 0:

    coeffs = np.array( res.args[0] )

    secondHump = objf( x3, coeffs )
    if secondHump > 1:
        coeffs /= secondHump
        xmax = x3
        xaux = x1
    else:
        xmax = x1
        xaux = x3
    ymax = float( y.subs( {x:xmax,
                           a:coeffs[0],
                           b:coeffs[1],
                           c:coeffs[2],
                           d:coeffs[3],
                           e:coeffs[4],
                           f:coeffs[5]} ) )
    yaux = float( y.subs( {x:xaux,
                           a:coeffs[0],
                           b:coeffs[1],
                           c:coeffs[2],
                           d:coeffs[3],
                           e:coeffs[4],
                           f:coeffs[5]} ) )

    print( coeffs )

    xs = np.linspace( 0, 1, 200 )
    plt.plot( xs, objf( xs, coeffs ) )

    print( "\n(x0/y0) = ({0:.2f}/{1:.2f}); "
           "(x1/y1) = ({2:.2f}/{3:.2f}); (x2/y2) = ({4:.2f}/{5:.2f});\n"
           "(x3/y3) = ({6:.2f}/{7:.2f}); (x4/y4) = ({8:.2f}/{9:.2f})"
           "".format( x0, float( y.subs( {x:x0,
                                          a:coeffs[0],
                                          b:coeffs[1],
                                          c:coeffs[2],
                                          d:coeffs[3],
                                          e:coeffs[4],
                                          f:coeffs[5]} ) ),
                      x1, float( y.subs( {x:x1,
                                          a:coeffs[0],
                                          b:coeffs[1],
                                          c:coeffs[2],
                                          d:coeffs[3],
                                          e:coeffs[4],
                                          f:coeffs[5]} ) ),
                      x2, float( y.subs( {x:x2,
                                          a:coeffs[0],
                                          b:coeffs[1],
                                          c:coeffs[2],
                                          d:coeffs[3],
                                          e:coeffs[4],
                                          f:coeffs[5]} ) ),
                      x3, float( y.subs( {x:x3,
                                          a:coeffs[0],
                                          b:coeffs[1],
                                          c:coeffs[2],
                                          d:coeffs[3],
                                          e:coeffs[4],
                                          f:coeffs[5]} ) ),
                      x4, float( y.subs( {x:x4,
                                          a:coeffs[0],
                                          b:coeffs[1],
                                          c:coeffs[2],
                                          d:coeffs[3],
                                          e:coeffs[4],
                                          f:coeffs[5]} ) ) ) )

    print( "\n\nAdjusted parameters:" )
    print( "x1 = {0:.2f}, x2 = {1:.2f}, x3 = {2:.2f}"
           "".format( x1, x2, x3 ) )

    print("\nRatio of maxima: 1 : {0:.2f}".format( yaux / ymax ) )

    intersections = sy.solve( y.subs( {a:coeffs[0],
                                       b:coeffs[1],
                                       c:coeffs[2],
                                       d:coeffs[3],
                                       e:coeffs[4],
                                       f:coeffs[5]} ) - yaux, x )

    keepList = []
    for i, xsec in enumerate( intersections ):
        if xsec >= 0.0 and xsec <= 1.0:
            keepList.append( i )
    intersections = np.array( intersections )[keepList]
    fraction = intersections[1] - intersections[0]
    print( "\nFraction of domain in winning area "
            "(above second maximum): {0:.1f}%"
            "".format( fraction * 100. ) )

    figureOfMerit = 1. - abs( 0.8 - yaux) * fraction
    print( "\nFigure of merit: {0:.2f}".format( figureOfMerit ) )

    if not iPython: plt.show()

else:

    print( "No solution in real domain" )



