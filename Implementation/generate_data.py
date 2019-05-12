import os

import numpy as np
import sympy
from sympy.utilities.lambdify import lambdify


def main():
    file_name = input("Enter file name: ")
    sym_fun_raw = input("Enter function form: ")
    sym_fun = sympy.sympify(sym_fun_raw)

    func = lambdify('x', sym_fun, "numpy")

    xx = np.linspace(-10, 10, num=21)
    yy = func(xx)

    currDir = os.path.dirname(__file__)
    file_name = os.path.join(currDir, "examples", file_name)

    f = open(file_name, "a+")
    f.write("# " + sym_fun_raw + "\n")

    for x, y in zip(xx, yy):
        f.write("%+f\t%+f\n" % (x, y))
    f.close()

    return

if __name__ == "__main__":
    main()
