import argparse

parser = argparse.ArgumentParser(description = "P norm of floating point numbers")
parser.add_argument('float_nums', type = float,nargs = '+', help = "A floating point number from the vector", metavar = 'f')
parser.add_argument('--p', type = int , help = "Power", default = 2)
args = parser.parse_args()

vec = args.float_nums

p = args.p
ans = 0
for num in vec:
    ans += abs(num)**p

ans = ans ** (1/p)

print("Norm of {a} is {b:.2f}".format(a = vec, b = ans))


