from fractions import Fraction
import os
import random
import string
import tabulate as tb
from itertools import product
import sympy as sy
import typer
import json
from pryttier.colors import coloredText, AnsiRGB, hsl2rgb

app = typer.Typer()

variablePath = os.path.abspath(os.path.join(".", "variables.json")) # to relative path
# Error fixed :)

with open(variablePath, "r") as f:
    try:
        variable: dict = json.load(f)
    except FileNotFoundError:
        print("Counldn't find the variable.json file")
    except PermissionError:
        print("Try reexcuting the programming with admin privileg")
    except (OSError, IOError) as e:
        print("Counldn't access the json due to an unkown error", e)

@app.command(help="Simplifies given expression.")
def simplify(expr: str):
    for i, j in variable.items():
        expr = expr.replace(i, f"({j})")
    res = sy.sympify(expr)
    sy.pprint(res.simplify())

@app.command(help="Evaluates given expression to floating point number.")
def evaluate(expr: str):
    for i, j in variable.items():
        expr = expr.replace(i, f"({j})")
    res = sy.sympify(expr)
    sy.pprint(res.evalf())

@app.command(help="Factorizes given expression.")
def factorize(expr: str):
    for i, j in variable.items():
        expr = expr.replace(i, f"({j})")
    res = sy.factor(expr)
    sy.pprint(res)

@app.command(help="Expands given expression.")
def expand(expr: str):
    for i, j in variable.items():
        expr = expr.replace(i, j)
    res = sy.expand(expr)
    sy.pprint(res)

@app.command(help="Solve For an expression in an equation.")
def solveFor(equation: str, solvefor: str):
    for i, j in variable.items():
        equation = equation.replace(i, f"({j})")
        solvefor = solvefor.replace(i, f"({j})")
    lhs, rhs = equation.split("=")
    eq = sy.Eq(sy.sympify(lhs), sy.sympify(rhs))
    res = sy.solve(eq, sy.sympify(solvefor))

    sy.pprint(res)

@app.command(help="Compute roots of a polynomial.")
def roots(expr: str):
    for i, j in variable.items():
        expr = expr.replace(i, f"({j})")
    res = sy.roots(expr)

    sy.pprint(list(res.keys()))

@app.command(help="Computes derivative.")
def derive(expr: str, syms: list[str]):
    for i, j in variable.items():
        expr = expr.replace(i, f"({j})")
    res = sy.diff(expr, *[sy.sympify(s) for s in syms])
    sy.pprint(res)

@app.command(help="Computes indefinite integral.")
def integrate(expr: str, syms: list[str]):
    for i, j in variable.items():
        expr = expr.replace(i, f"({j})")
    res = sy.integrate(expr, *[sy.sympify(s) for s in syms])
    sy.pprint(res)

@app.command(help="Computes definite integral.")
def integrate_def(expr: str, syms: str, lims: str):
    syms = syms.split(" ")
    for i, j in variable.items():
        expr = expr.replace(i, f"({j})")
    bounds = [(l.split("->")[0], l.split("->")[1]) for l in lims.split(" ")]
    if len(bounds) == 1:
        fvars = []
        for s in range(len(syms)):
            if s == len(syms) - 1:
                fvars.append((sy.sympify(syms[s]), bounds[0][0], bounds[0][1]))
            else:
                fvars.append(sy.sympify(syms[s]))
        res = sy.integrate(expr, *fvars)
    else:
        try:
            res = sy.integrate(expr, *[(sy.sympify(syms[s]), bounds[s][0], bounds[s][1]) for s in range(len(syms))])
        except IndexError:
            print("Either specify limits per integration or specify one limit for final evaluation")
            return -1
    sy.pprint(res)

@app.command(help="Generate Truth table for given boolean expression.")
def truthtable(expr: str):
    for i, j in variable.items():
        expr = expr.replace(i, j)
    chars = string.ascii_lowercase
    res = sy.simplify(sy.sympify(expr))
    print(f"Simplified: {res}")
    symbols = list(res.free_symbols)
    symbols.sort(key=lambda v: chars.index(str(v)))

    colStep = Fraction(360, len(symbols)) 
    syms = [[coloredText(str(a), AnsiRGB(hsl2rgb((i*colStep, 100, 80)))) for i, a in enumerate(symbols)] + ["o"]]
    ins = list(product(*[[0, 1] for _ in range(len(symbols))]))

    func = sy.lambdify(symbols, res)

    io = [[*a, int(func(*a))] for a in ins]
    offColor = (252, 98, 85)
    onColor = (88, 196, 221)
    cio = [[coloredText(str(b), AnsiRGB(onColor if b == 1 else offColor)) for b in a] for a in io]

    table = tb.tabulate(syms + cio, tablefmt="outline")
    print(table)

@app.command(help="Unit Conversion")
def convert(utype: str, u1: str, u2: str, number: int):
    res = 0
    prefixes = {
        "G": 1e9,
        "M": 1e6,
        "k": 1000,
        "h": 100,
        "da": 10,
        "": 1,
        "d": 0.1,
        "c": 0.01,
        "m": 0.001,
        "µ": 1e-6,
        "n": 1e-9,
    }
    specialLengths = {
        "in": 0.0254,
        "ft": 0.3048,
        "yd": 0.9144,
        "mi": 1609.34,
        "ly": 9460660000000000,
    }
    specialAreas = {
        "sqin": 0.00064516,
        "sqft": 0.092903,
        "sqyd": 0.836127,
        "sqmi": 2.59e+6,
        "ha": 10000,
        "ac": 4046.86,
    }
    specialVolumes = {
        "l": 0.001,
        "ml": 1e-6,
        "cbft": 0.0283168,
        "cbin": 1.6387e-5,
        "ha": 10000,
        "ac": 4046.86,
    }
    specialMass = {
        "t": 1e+6,
        "lb": 453.592,
        "oz": 28.3495,
        "ust": 907185,
        "impt": 1.016e+6
    }

    if utype.lower() == "length":
        lengthUnits = {}
        for i, j in prefixes.items():
            lengthUnits[f"{i}m"] = j
        for i, j in specialLengths.items():
            lengthUnits[i] = j

        ratio1 = lengthUnits[u1]
        ratio2 = lengthUnits[u2]

        res = Fraction(ratio1, ratio2)

    if utype.lower() == "area":
        areaUnits = {}
        for i, j in prefixes.items():
            areaUnits[f"sq{i}m"] = j**2
        for i, j in specialAreas.items():
            areaUnits[i] = j

        ratio1 = areaUnits[u1]
        ratio2 = areaUnits[u2]

        res = Fraction(ratio1, ratio2)

    if utype.lower() == "volume":
        volumeUnits = {}
        for i, j in prefixes.items():
            volumeUnits[f"cb{i}m"] = j**3
        for i, j in specialVolumes.items():
            volumeUnits[i] = j

        ratio1 = volumeUnits[u1]
        ratio2 = volumeUnits[u2]

        res = Fraction(ratio1, ratio2)

    if utype.lower() == "mass":
        massUnits = {}
        for i, j in prefixes.items():
            massUnits[f"{i}g"] = j
        for i, j in specialMass.items():
            massUnits[i] = j

        ratio1 = massUnits[u1]
        ratio2 = massUnits[u2]

        res = Fraction(ratio1, ratio2)

    if utype.lower() in ["freq" or "frequency"]:
        freqUnits = {}
        for i, j in prefixes.items():
            freqUnits[f"{i}Hz"] = j

        ratio1 = freqUnits[u1]
        ratio2 = freqUnits[u2]

        res = Fraction(ratio1, ratio2)

    res *= number
    print(res)

@app.command(help="Random dice roll")
def roll(n: list[int] = typer.Argument(..., help="List of dice")):
    rans = []
    for i in n:
        ran = random.randint(0, i)
        rans.append(ran)
    print(*rans)
    print(f"Total: {sum(rans)}")
    print(f"Max: {max(rans)}")
    print(f"Min: {min(rans)}")

@app.command(help="Define a variable")
def defvar(var_name: str, value: str):
    variable[var_name] = value
    with open(variablePath, "w") as f:
        json.dump(variable, f, indent=4)

@app.command(help="Remove a variable")
def rmvar(var_name: str):
    try:
        variable.pop(var_name)
    except KeyError:
        print(f"No Variable named {var_name}")
    with open(variablePath, "w") as f:
        json.dump(variable, f, indent=4)

@app.command(help="Prints all defined variables")
def pvars():
    for i, j in variable.items():
        sy.pprint(f"{i}: {sy.sympify(j)}")

@app.command(help="Clear all variable definitions")
def clv():
    stack = {}
    with open(variablePath, "w") as f:
        json.dump(stack, f, indent=4)

if __name__ == "__main__":
    app()