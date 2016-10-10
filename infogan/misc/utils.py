from __future__ import print_function
from __future__ import absolute_import
import errno
import os


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def pshape(string,x):
    if isinstance(x,list):
        print ("Shape: of '%s' is-len  %s " % (string,len(x)))
    else:
        print ("Shape: of '%s' is %s " % (string,x.shape,))
	

def pstr(string,x):
	print ("STR: '%s' is %s " % (string,x,))

def pstr2(x):
    print ("STR: '%s' is %s " % (str(x),x,))

def pstrall(string,x):
    for s in x:
        print ("STR: '%s' is %s " % (string,s,))

def pall(string,x):
	pstr(string,x)
	pshape(string,x)

