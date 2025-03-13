from src.dsl.arc_types import *
from src.dsl.constants import *
from src.dsl.dsl import *
from src.utils import ROOT
from src.dsl.main import get_data

from viztracer import log_sparse, VizTracer
from functools import wraps
import inspect
import json
import tempfile

key = "e9afcf9a"
ex = get_data()["train"][key][0]

I = ex["input"]

tracer = VizTracer(log_func_args=True, log_func_retval=True, log_print=True)
tracer.start()


@log_sparse(stack_depth=10)
def solve_e9afcf9a(I):
    x1 = astuple(TWO, ONE)
    x2 = crop(I, ORIGIN, x1)
    x3 = hmirror(x2)
    x4 = hconcat(x2, x3)
    x5 = hconcat(x4, x4)
    O = hconcat(x5, x4)

    for name, value in locals().items():
        tracer.log_var(name, value)
    return O


O = solve_e9afcf9a(I)
assert O == ex["output"]
tracer.stop()

# Create a temporary file that gets automatically cleaned up
with tempfile.NamedTemporaryFile(suffix=".json") as tmpf:
    tracer.save(tmpf.name)

    # Read and print the trace
    with open(tmpf.name, "r") as f:
        trace_data = json.load(f)
        var_events = list(
            filter(
                lambda d: d.get("cat") == "INSTANT",
                trace_data["traceEvents"],
            )
        )
        varlocals = {}
        for e in var_events:
            varlocals[e["name"]] = eval(e["args"]["object"])
        debug(varlocals)
