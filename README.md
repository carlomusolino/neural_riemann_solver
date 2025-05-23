# Neural Riemann Solver for 1D Relativistic Hydrodynamics

This repository contains the code presented in the paper
**"A Neural Riemann Solver for Relativistic Hydrodynamics"**.

Install the package in editable mode with:

```bash
pip install -e ./
```

---

## Code Structure

The package is organized under the `riemannML/` directory:

* `riemannML.data` — data generation routines for training;
* `riemannML.exact` — analytical Riemann solver components;
* `riemannML.solvers` — neural network models and the neural Riemann solver;
* `riemannML.hydro` — hydrodynamics solvers and approximate Riemann solvers;
* `riemannML.training_utils` — training and evaluation scripts;
* `riemannML.utilities` — numerical helper functions.

---

## Reproducing Results

The main entry points are in the `scripts/` directory.

### Train the main model (used in the paper)

To train the main solver model, run:

```bash
python train_models_script.py --seed 42 --n_neurons 64 --n_dsets 131072 --batch 128 --n_layers 2
```

This produces model weights in:

```
models/2_layers_64_neurons/131072_samples/42/
```

The double shock network is saved at:

```
${MODEL_DIR}/d_shock/checkpoints/best_model.pt
```

You can verify it using the provided SHA256 checksum:

```bash
sha256sum -c model_checksum.sha256
```

However, note that the results presented in the paper were obtained
on an AMD Mi50 platform with `rocm` version `6.2.1` and `pyTorch` version `2.5.1`.
Results might be slightly different on different platforms using different software.


To instantiate a full neural Riemann solver, use:

```python
from riemannML.solvers.solver import construct_riemann_solver
```

---

### Running Shock Tube Tests

Training automatically runs first-order simulations for the four shock tube problems described in the paper on a grid of 800 points. Results are saved in:

```
${MODEL_DIR}/prob_#/fo/
```

Each test directory contains:

* `errors.pkl` — L1 error on density,
* `results.pkl` — simulation output,
* `timers.pkl` — timing information.

---

### Running the Full Test Suite

To run a resolution study and second-order WENO tests:

```bash
python run_sims_script.py --seed <SEED> --n_neurons <N_NEURONS> \
    --n_dsets <N_DSETS> --n_layers <N_LAYERS> --basedir <BDIR>
```

**Example:**

```bash
python run_sims_script.py --seed 42 --n_neurons 64 \
    --n_dsets 131072 --n_layers 2 --basedir models
```

---

### Ensemble Models (Appendix)

To run tests using an ensemble of models (as described in the paper's appendix):

```bash
python run_sims_ensemble_script.py --seed_combination <SEED1,SEED2,...> \
    --n_neurons <N_NEURONS> --n_dsets <N_DSETS> \
    --n_layers <N_LAYERS> --basedir <BDIR>
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE.md) file for details.


## Bug reporting 

Please report any bugs or issues to Carlo Musolino at [musolino@itp.uni-frankfurt.de](mailto:musolino@itp.uni-frankfurt.de)