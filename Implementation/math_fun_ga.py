import random

import matplotlib.pyplot as plt
import numpy as np
import sympy
from deap import algorithms, base, creator, tools
from sympy.utilities.lambdify import lambdify


class MathFunGA:

    # CXPB is the probability with which two individuals are crossed
    CXPB = 0.5
    # MUTPB is the probability for mutating an individual
    MUTPB = 0.3
    # Threshold for stopping the evolution once this fitness is reached
    FITNESS_TARGET = 1E-7
    # Starting population size
    STARTING_POPULATION = 2000
    # Range from which random values are drawn for the starting population.
    STARTING_MIN_VALUES = -10
    STARTING_MAX_VALUES = +10

    def __init__(self, data_path, sym_fun_raw, NGEN):
        self.NGEN = NGEN

        self.original_points_X, self.original_points_Y, self.original_fun = MathFunGA.loadData(data_path)
        self.sym_fun, self.free_syms, self.funForTitle = MathFunGA.cleanUpFunction(sym_fun_raw)

        xMin = self.original_points_X.min()
        xMax = self.original_points_X.max()
        width = xMax-xMin+1
        xCoordsLinear = np.linspace(xMin - width/25, xMax + width/25, width*30)
        self.xCoordsPlot = np.union1d(xCoordsLinear, self.original_points_X)

        self.figure = plt.figure(figsize=(10,7))
        self.ax = self.figure.add_subplot(1,1,1)

        self.toolbox = self.prepareAlgorithm()
        self.population = self.toolbox.population(n=MathFunGA.STARTING_POPULATION)

    def prepareAlgorithm(self):
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,))     # negative weight
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Attribute generator
        toolbox.register("attr_bool", random.uniform, 
                MathFunGA.STARTING_MIN_VALUES, MathFunGA.STARTING_MAX_VALUES)
        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                toolbox.attr_bool, n = len(self.free_syms)-1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.fitnessFun)
        toolbox.register("mate", tools.cxBlend, alpha=0.2)
        # toolbox.register("mate", tools.cxSimulatedBinary, eta=3)

        # The independent probability of each attribute to be mutated 'indpb'
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.15, indpb=0.4)
        toolbox.register("select", tools.selTournament, tournsize=3)

        return toolbox

    def run(self):
        breakEarly = False
        for gen in range(1, self.NGEN+1):
            try:
                if breakEarly:
                    print("Stopped by user!")
                    break
                offsprings = algorithms.varAnd(self.population, self.toolbox, 
                        cxpb=MathFunGA.CXPB, mutpb=MathFunGA.MUTPB)
                fitnesses = self.toolbox.map(self.toolbox.evaluate, offsprings)
                for fit, ind in zip(fitnesses, offsprings):
                    ind.fitness.values = fit
                self.population = self.toolbox.select(offsprings, k=len(self.population))

                theBestThisPop = tools.selBest(self.population, k=1)[0]
                theBestFitness = theBestThisPop.fitness.getValues()[0]
                print((str(gen) + ":").ljust(5) + str(theBestFitness))
                if theBestFitness < MathFunGA.FITNESS_TARGET:
                    print("Satisfying fitness reached!")
                    break
            except KeyboardInterrupt:
                breakEarly = True
        else:
            print("Maximum number of iterations reached!")

        self.showResults()
       

    def showResults(self):
        # Uncomment to plot all solutions of the final population.
        # for ind in self.population:
            # self.plot(ind)
        
        theBest = tools.selBest(self.population, k=1)[0]
        maxFit = theBest.fitness.getValues()[0]
        dictOfResults = dict(zip(self.free_syms[1:], theBest))
        calculatedFunAsStr = self.funForTitle.subs(dictOfResults)

        print("\nCalculated results:")
        for k,v in dictOfResults.items():
            print(k + " = " + str(v))
        print("\nFinal function:")
        print(calculatedFunAsStr)
        print("\nFitness: " + str(maxFit))

        self.ax.title.set_text(calculatedFunAsStr)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.grid(True)

        self.plot(theBest)

        if self.original_fun:
            self.ax.plot(self.xCoordsPlot, self.original_fun(self.xCoordsPlot), 'g:')

        self.ax.plot(self.original_points_X, self.original_points_Y, 'ro')
        plt.show()

    def plot(self, individual):
        yCoords = self.sym_fun(self.xCoordsPlot, *individual)
        self.ax.plot(self.xCoordsPlot, yCoords)

    def fitnessFun(self, individual):
        values = self.sym_fun(self.original_points_X, *individual)
        squares = (values - self.original_points_Y) ** 2
        return squares.sum(),   # Comma is crucial!

    @staticmethod
    def loadData(path):
        with open(path) as f:
            lines = [line.strip() for line in f]
        original_fun = next((x for x in lines if x and x[0]=="#"), None)
        func = None
        if original_fun != None:
            original_fun = original_fun[1:].strip()
            func, free_syms_str, _ = MathFunGA.cleanUpFunction(original_fun)
            if len(free_syms_str) != 1:
                raise ValueError("Function must not have any variables other than 'x'.")
        # Remove empty (blank) lines and comments
        lines = list(filter(
            lambda x: x and x[0] != "#" and not x.startswith("//"),
            lines))

        points_X = np.zeros(len(lines))
        points_Y = np.zeros(len(lines))
        for index, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 2:
                raise ValueError("Cannot parse: " + line)
            points_X[index] = float(parts[0])
            points_Y[index] = float(parts[1])
        return points_X, points_Y, func

    @staticmethod
    def cleanUpFunction(raw_fun):
        sym_fun = sympy.sympify(raw_fun)
        free_syms = sym_fun.free_symbols
        if sympy.symbols('x') not in free_syms:
            raise ValueError("Function must have an 'x' variable.")
        free_syms -= { sympy.symbols('x') }
        free_syms_str = sorted([str(sym) for sym in free_syms])
        free_syms_str.insert(0, 'x')
        # Returns a numpy-ready function
        func = lambdify(free_syms_str, sym_fun, "numpy")
        return func, free_syms_str, sym_fun


def main():
    print("STARTING EXECUTION\n")

    data_path = input("Path to input data: ")
    sym_fun_raw = input("Enter function form: ")
    NGEN = int(input("Number of iterations: "))

    # NGEN = 200

    # data_path = "examples/sinExpNarrow.txt"
    # sym_fun_raw = "A*sin(B*x+C)*exp(D*x+F)"

    # data_path = "examples/sinExpNarrow.txt"
    # sym_fun_raw = "A*x^3+B*x^2+C*x+D"

    # data_path = "examples/sinExpWide.txt"
    # sym_fun_raw = "A*sin(B*x+C)*exp(D*x+F)"

    # data_path = "examples/gauss.txt"
    # sym_fun_raw = "1/(sqrt(2*pi)*SIGMA)*exp(-(x-MI)^2/(2*SIGMA^2))"

    # data_path = "examples/gaussError.txt"
    # sym_fun_raw = "1/(sqrt(2*pi)*SIGMA)*exp(-(x-MI)^2/(2*SIGMA^2))"

    # data_path = "examples/pureSin.txt"
    # sym_fun_raw = "A+B*sin(C*x+D)"

    # data_path = "examples/sqrtAbs.txt"
    # sym_fun_raw = "A+B*sqrt(abs(C*x+D))"

    # data_path = "examples/sqrtAbs2.txt"
    # sym_fun_raw = "A+B*sqrt(abs(C*x+D))"

    # data_path = "examples/sqrtAbs.txt"
    # sym_fun_raw = "sqrt(abs(x))"                  # there can be 0, 1, or more variables

    # data_path = "examples/randomData.txt"
    # sym_fun_raw = "AH*sin(BU*x+CR)*exp(D*x+F)"    # variables do not have to be symbols

    MathFunGA(data_path, sym_fun_raw, NGEN).run()


if __name__ == "__main__":
    main()
