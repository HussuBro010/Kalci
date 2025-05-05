import random
import tabulate as tb
from itertools import product
import sympy as sy
import typer
import json
from pryttier.colors import coloredText, AnsiRGB, hsl2rgb

app = typer.Typer()

stackPath = "C:/Users/DELL/Desktop/Hussain/Programming/Python/Python/Tools/kalci/variables.json"

with open(stackPath, "r") as f:
    stack: dict = json.load(f)

@app.command(help="Simplifies given expression.")
def simplify(expr: str):
    for i, j in stack.items():
        expr = expr.replace(i, j)
    res = sy.simplify(expr)
    sy.pprint(res)

@app.command(help="Factorizes given expression.")
def factorize(expr: str):
    for i, j in stack.items():
        expr = expr.replace(i, j)
    res = sy.factor(expr)
    sy.pprint(res)

@app.command(help="Expands given expression.")
def expand(expr: str):
    for i, j in stack.items():
        expr = expr.replace(i, j)
    res = sy.expand(expr)
    sy.pprint(res)

@app.command(help="Solve For en expression in an expression.")
def solveFor(equation: str, solvefor: str):
    for i, j in stack.items():
        equation = equation.replace(i, j)
        solvefor = solvefor.replace(i, j)
    lhs, rhs = equation.split("=")
    eq = sy.Equality(sy.sympify(lhs), sy.sympify(rhs))
    res = sy.solve(eq, sy.sympify(solvefor))

    sy.pprint(res)

@app.command(help="Compute roots of a polynomial.")
def roots(expr: str):
    for i, j in stack.items():
        expr = expr.replace(i, j)
    res = sy.roots(expr)

    sy.pprint(res)

@app.command(help="Computes derivative.")
def derive(expr: str, syms: list[str]):
    for i, j in stack.items():
        expr = expr.replace(i, j)
    res = sy.diff(expr, *[sy.sympify(s) for s in syms])
    sy.pprint(res)

@app.command(help="Computes indefinite integral.")
def integrate(expr: str, sym: list[str]):
    for i, j in stack.items():
        expr = expr.replace(i, j)
    res = sy.integrate(expr, *[sy.sympify(s) for s in syms])
    sy.pprint(res)

@app.command(help="Computes definite integral.")
def integrate_def(expr: str, syms: list[str], lims: str):
    for i, j in stack.items():
        expr = expr.replace(i, j)
    lims = [(l.split("->")[0], l.split("->")[1]) for l in lims.split(" ")]
    if len(lims) == 1:
        res = sy.integrate(expr, *[(sy.sympify(syms[s]), lims[0][0], lims[0][1]) for s in range(len(syms))])
    else:
        try:
            res = sy.integrate(expr, *[(sy.sympify(syms[s]), lims[s][0], lims[s][1]) for s in range(len(syms))])
        except IndexError:
            print("Either specify limits per integration or specify one limit for all")
            return -1
    sy.pprint(res)

@app.command(help="Generate Truth table for given boolean expression.")
def truthtable(expr: str):
    for i, j in stack.items():
        expr = expr.replace(i, j)
    chars = "abcdefghijklmnopqrstuvwxyz"
    res = sy.simplify(sy.sympify(expr))
    print(f"Simplified: {res}")
    symbols = list(res.free_symbols)
    symbols.sort(key=lambda v: chars.index(str(v)))

    colStep = 360/len(symbols)
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

        res = ratio1/ratio2

    if utype.lower() == "area":
        areaUnits = {}
        for i, j in prefixes.items():
            areaUnits[f"sq{i}m"] = j**2
        for i, j in specialAreas.items():
            areaUnits[i] = j

        ratio1 = areaUnits[u1]
        ratio2 = areaUnits[u2]

        res = ratio1/ratio2

    if utype.lower() == "volume":
        volumeUnits = {}
        for i, j in prefixes.items():
            volumeUnits[f"cb{i}m"] = j**3
        for i, j in specialVolumes.items():
            volumeUnits[i] = j

        ratio1 = volumeUnits[u1]
        ratio2 = volumeUnits[u2]

        res = ratio1/ratio2

    if utype.lower() == "mass":
        massUnits = {}
        for i, j in prefixes.items():
            massUnits[f"{i}g"] = j
        for i, j in specialMass.items():
            massUnits[i] = j

        ratio1 = massUnits[u1]
        ratio2 = massUnits[u2]

        res = ratio1/ratio2

    if utype.lower() in ["freq" or "frequency"]:
        freqUnits = {}
        for i, j in prefixes.items():
            freqUnits[f"{i}Hz"] = j

        ratio1 = freqUnits[u1]
        ratio2 = freqUnits[u2]

        res = ratio1/ratio2

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
    stack[var_name] = value
    with open(stackPath, "w") as f:
        json.dump(stack, f, indent=4)

@app.command(help="Remove a variable")
def rmvar(var_name: str):
    try:
        stack.pop(var_name)
    except KeyError:
        print(f"No Variable named {var_name}")
    with open(stackPath, "w") as f:
        json.dump(stack, f, indent=4)

@app.command(help="Prints all defined variables")
def pvars():
    for i, j in stack.items():
        print(f"{i}: {j}")

@app.command(help="Clear all variable definitions")
def clv():
    stack = {}
    with open(stackPath, "w") as f:
        json.dump(stack, f, indent=4)

if __name__ == "__main__":
    app()