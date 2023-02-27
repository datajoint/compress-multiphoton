import numpy as np
from time import time
import plotly.graph_objects as go
from .compression import compute_quantal_size


def timer_func(func):
    # This function shows the execution time of the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


def analyze(scan):
    qs = compute_quantal_size(scan)

    print(f"Quantal size: {quantal_size}")
    print(f"Intercept: {zero_level}")

    x = np.arange(len(qs["unique_pixels"]))
    f = go.Figure(go.Scatter(x=qs["unique_pixels"], y=qs["unique_variances"], mode="markers"))
    f.add_trace(go.Scatter(x=x, y=qs["model"].predict(x.reshape(-1, 1))))
    f.update_layout(
        width=700,
        height=700,
        yaxis=dict(title_text="Variance"),
        xaxis=dict(title_text="Pixel Intensity"),
        showlegend=False,
    )

    return f
