import os
import sys

if __name__ == '__main__':
    port = 8097
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    result = os.system("python -m visdom.server -port %i" % port)