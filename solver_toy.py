from arc_types import *
from constants import *
from dsl import *
from src.utils import ROOT

key = "e9afcf9a"


def solve_e9afcf9a(I):
    x1 = astuple(TWO, ONE)
    x2 = crop(I, ORIGIN, x1)
    x3 = hmirror(x2)
    x4 = hconcat(x2, x3)
    x5 = hconcat(x4, x4)
    O = hconcat(x5, x4)
    # !viztracer: log_var("x1", x1), log_var("x2", x2), log_var("x3", x3), log_var("x4", x4), log_var("x5", x5), log_var("O", O), log_var("I", I)
    return O


I = (ROOT / f"data/test/{key}.json".format(key)).read_json()
