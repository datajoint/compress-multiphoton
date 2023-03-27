import numpy as np
from time import time
import plotly.graph_objects as go
from .compression import compute_sensitivity


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
    qs = compute_sensitivity(scan)

    print(f'Quantal size: {qs["sensitivity"]}')
    print(f'Intercept: {qs["zero_level"]}')

    x = np.arange(qs["min_intensity"], qs["max_intensity"])
    f = go.Figure(go.Scatter(x=x, y=qs["variance"], mode="markers"))
    f.add_trace(go.Scatter(x=x, y=qs["model"].predict(x.reshape(-1, 1))))
    f.update_layout(
        width=700,
        height=700,
        yaxis=dict(title_text="Variance"),
        xaxis=dict(title_text="Intensity Levels"),
        showlegend=False,
    )

    return f
