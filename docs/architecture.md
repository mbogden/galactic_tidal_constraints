# Architecture

> **Status: this is the plan, and it's still in testing and development.** Built so far: the
> Simulator and Image Creator stages and the `simr` workflow that runs them (see the module docs in
> [modules/](modules/)). The metrics stage, the applications and the optimizer are still plans, and details may
> change as they're written. Decisions and rules are in [PLAN.md](../PLAN.md).

This project simulates galaxy collisions, turns them into images, and scores how well they match a
target. This page describes how the code is organized to do that: its levels, and what passes
between them.

## Levels
| Level | What it is | Runs on |
|---|---|---|
| **Experiment** | One question ("Can X do Y?"), answered with many runs plus analysis between them | Slurm cluster, usually one run per node |
| **Application / run** | The application is the code you launch; a run is one launch with one target and one config | One node, main process |
| **Workflow** | Everything one process does for its batch, including disk reads and writes | One process |
| **Stage** | One science step: Simulator, Image Creator or Metrics | Same process |

"Pipeline" isn't a structural term here. Say "simulation" or "stage" where "SPAM run" would be
unclear.

## How a run works
1. An experiment launches a run with one target and one config.
2. The application expands the config into tables of models and stage settings.
3. The application splits the work and sends each process a config dict for its slice.
4. Each process runs a workflow, which calls the stages and reads or writes files as needed.
5. Each workflow returns a metric array, and the application joins them into the run's metric array.
6. The run saves its tables and metrics, and the experiment analyzes results across runs.

## Experiment
- Changes the config for each run. Each run is parallelized through Slurm.
- Answers its question with data visualization and analysis between runs.
- Data lives in the experiment's own folder, with a subfolder per run. Data shared between
  experiments may move to a larger shared folder later.

## Application and run
- TNG, GZ comparison, DL training and Target GA each get their own application, and the code lives
  in the repo.
- The application expands the config into model, simulator, image and metric tables.
- The run keeps those tables and its final metric array in its data folder.
- The application chooses how to split the work across processes. Shrinking batches are one example,
  and suit SPAM, where a simulation can take from a second to over a minute.
- Composite scores, such as weighted averages of metrics, are computed here from the returned
  array, so weights can change without re-simulating. A combination that needs the arrays themselves,
  not just the metric values, is a normal metric method.

## SIMR module and workflow
- `simr` is a module with an optional main function. It holds the workflow and handles disk access.
- Each simulation is reused for all its images and metrics, because the simulation is the expensive
  part.
- SIMR is in charge of disk reads and writes, including saving and resuming, using one standard
  per-model folder structure.
- Files are flexible per experiment: none (GA fine-tuning), permanent (GZ Merger models) or scratch
  (high-resolution particles to re-visualize later). An application can also take returned data and
  save it its own way.
- A failure returns `NaN` and writes the error to the log, so one bad model doesn't stop the run.
- SIMR's own pool is simple and optional. Higher-level code can run its own and call lower-level
  code directly.

## Stages
A stage does one thing per call, takes data in and returns data out, and never touches disk or
parallelization.

| Stage | Input | Output | Built |
|---|---|---|---|
| **Simulator** | One model's encounter parameters | An orbit or particles | `simr/simulation.py` ([modules/simulation.md](modules/simulation.md), [spam_physics.md](spam_physics.md)) |
| **Image Creator** | One particle set, plus its model (which sets the frame) | One image, `float32` in [0, 1] | `simr/image_creator.py` ([modules/image_creator.md](modules/image_creator.md)) |
| **Metrics** | An image or orbit, plus the target | One float, or a dict for descriptive metrics | Not yet |

- SPAM stays replaceable behind the Simulator stage.
- The Image Creator is target-free: its config sets the pixels the two galaxies' centers land on, and
  higher-level code decides whether those come from a target image. The same frame maps an orbit into
  the image's pixels.
- The metrics stage holds several methods, including orbit and image comparison.

## Models and stage lists
- Encounter parameters are the one hard input every run needs. They're their own input, a list of
  models kept apart from the config: the models say what to simulate, the config says how.
- Each stage can have one entry or a list. A list can appear only directly under a stage; anything
  deeper is an ordinary value, like an image size.
- Lists combine as every combination, and higher-level code keeps that to a sensible size.
- A list's index is its entry's name.

For example, two models with this config:
```json
{
  "simulator": [ {...} ],
  "image":     [ {...}, {...} ],
  "img_metrics": [ {...}, {...} ]
}
```
How to write a config today, with real examples: [configuration.md](configuration.md).

## Metric arrays: two levels
SIMR and the workflow return the metrics as full arrays, with no automatic reduction, so batches
always join cleanly along the model axis. There are two levels, so nothing is copied, NaN-padded or
counted twice:
```
sim_metrics[model, simulator, entry]          metrics whose inputs are sim.* items (orbits, particles)
img_metrics[model, simulator, image, entry]   metrics with any img.* input
```
- The example above returns `img_metrics` of shape `(2, 1, 2, 2)`.
- A single entry is an axis of length 1. Application and analysis code reduce the arrays where they
  use them.
- Workflows can still send other float arrays where needed.
