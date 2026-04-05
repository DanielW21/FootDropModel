# FootDropModel

FootDropModel is a small gait simulation project for studying ankle motion, tibialis anterior stimulation, and foot-clearance behavior during walking. The code builds a reference gait, runs a muscle-driven ankle simulation, and then visualizes or optimizes the stimulation profile used in swing.

## How It Works

The project starts with a kinematic gait representation, turns that into simulation-ready joint and foot trajectories, and then feeds those trajectories into a muscle-based (TA and Soleus) ankle model. The simulation combines muscle activation dynamics, joint limits, gravity, and floor contact checks to estimate how the foot behaves through a gait cycle.


## Key Scripts

### `src/scripts/normal_gait_generation.py`

This is the entry point for generating the gait data used by the rest of the project. It builds the reference pose sequence through `GaitEngine`, then saves CSV files into `output/sim_data/` for later simulation and replay.

### `src/sims/muscle_sim.py`

This is the core simulation module. It loads the gait baseline, runs the coupled ankle and muscle dynamics, applies TA stimulation during swing, and returns the simulated motion and activation results.

### `src/scripts/foot_drop_FES_gait_optimization.py`

This script reconstructs a stimulation profile from optimized nodes, runs the simulation with that profile, and opens the animation.


## Project Layout

- `configs/` contains the model configuration.
- `src/muscles/` contains the muscle model.
- `src/sims/` contains the main simulation code.
- `src/scripts/` contains runnable entry points and viewers.
- `src/optimization/` contains the optimization routines for stimulation profiles.
