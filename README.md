## Bernstein splines solvers for Hilfer derivative fractional differential equations

![A phase portrait heart<3](docs/pictures/stable_heart_empty.png)

[Update: check out the arXiv preprint outlining the method and convergence for IVP's!](https://doi.org/10.48550/arXiv.2503.22335)

This repository implements the Bernstein splines based methods as developed from my [MSc. thesis](https://repository.tudelft.nl/record/uuid:38ca4f2d-a0ec-4bde-b871-0c9e6d5b3f8a) on fractional-order differential equations with Hilfer fractional derivatives. Both a "local" and "global" setup are implemented in ```fr.splines.SplineSolver```, where the local setup iterates knot for knot and the global method iterates the equation on the whole time domain at once. The global setup has as advantage that nonlocal BVP-type constraints can be handled directly. The global setup however requires more stringent convergence requirements, which are near-unconditionally satisfied in the local approach for most knot sizes. All solvers are implemented using the Hilfer fractional derivative, for which parameter $\beta \in [0,1]$ can be provided through ```beta_vals```. Here, $\beta = 1$ gives the familiar Caputo fractional derivative, and $\beta = 0$ the Riemann-Liouville fractional derivative. For more information on the Hilfer derivative and its connection to IVP-solutions, see for instance [(Furati, 2012)](https://www.sciencedirect.com/science/article/pii/S0898122112000193).
Finally, we remark that the method can be used as a powerful ODE-solver by choosing non-fractional orders $\alpha \in \mathbb{N}$.

## Installation Windows:

```
python -m venv .venv ; .venv\Scripts\activate.bat ; python -m pip install -e .
```

## Installation Linux:

```
python -m venv .venv ; source .venv/bin/activate ; python -m pip install -e .
```

## Optional: CUDA acceleration
Fracnum supports using cupy as a drop-in replacement for numpy. To enable this, install optional dependences with `pip install -e .[gpu]`, and change `BACKEND_ENGINE=cupy` in `environment.env`.  

`cupy` is not faster than `numpy` for smaller problems. But [experimentally](experiments/performance/running_times.txt), at a simulation of about 750.000 iterations, `cupy` saves significant time.

NOTE: `cupy` support with analytical `mpmath` sinusoidal forcing through `forcing_params` is not yet implemented. The best workaround is to pass sine forcing to the DE function itself using splines, adding a slight approximation error.

## Example file for pretty pictures:

```
python examples/VdP/frac_vdp_oscillator_example.py
```

## Example experimentation file:

```
python examples/VdP/frac_vdp_oscillator_experimentation.py
```
