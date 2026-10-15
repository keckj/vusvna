import argparse

import vusvna

parser = argparse.ArgumentParser(prog="python -m vusvna", description=vusvna.__doc__.strip().split("\n")[0])
parser.add_argument("--include", action="store_true", help="print the include directory of the Slang module")
parser.add_argument("--version", action="version", version=vusvna.__version__)
args = parser.parse_args()
if args.include:
    print(vusvna.get_include())
else:
    parser.print_help()
