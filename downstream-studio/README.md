# Downstream Studio

Downstream Studio uses a React interface with a local Python API and experiment runner. Existing projects and runs remain in the configured workspace directory.

## Frontend development

The React source lives in `frontend/`. Build it into the Python package with:

```bash
cd frontend
npm install
npm run build
```

The Python server serves the compiled interface from `src/downstream_studio/web/`.

A project-based web interface for configuring, running, and interpreting
downstream probes on biological interface embeddings.

## Features

- Persistent projects, datasets, experiments, and run artifacts.
- Project-level **Data & artifacts**, **Train**, and **Experiment history** workspaces.
- Register large local/HPC data by path, directory, or glob pattern.
- Upload target or embedding CSV files through the browser.
- Automatic embedding-model, dimension, and numeric-target discovery.
- Select embedding models and target fields independently.
- Linear and configurable MLP architectures with a visual preview.
- Visual MLP builder with 1–12 hidden layers, per-layer neuron counts, and
  `ReLU`, `GELU`, `SiLU`, `Tanh`, `Leaky ReLU`, or identity activations.
- Adjustable linear bottleneck width. Zero is direct regression; a positive
  width adds an activation-free linear projection and remains a linear probe.
- Configure splits, epoch checkpoints, optimizer parameters, seed, and device.
- Quick, balanced, and thorough experiment presets.
- Background execution with completed-evaluation progress, elapsed time,
  estimated remaining time, live logs, and cancellation.
- Filterable results by target, probe, split, and embedding model, with R²
  curves, leaderboards, prediction scatter, residual plots, throughput, and
  downloadable CSV artifacts.
- Experiment history with open, reuse-as-new, and recoverable delete actions.
- SQLite metadata and filesystem artifacts; no external service is required.

## Local installation

Install the analysis engine and Studio in the same Python environment:

```bash
cd /Users/divar/Documents/biotech/downstream-analysis
python3 -m pip install -e '.[csv]'

cd /Users/divar/Documents/biotech/downstream-studio
python3 -m pip install -e .
```

Start the application:

```bash
downstream-studio \
  --workspace /Users/divar/Documents/biotech/downstream-studio-data \
  --browse-root /Users/divar/Documents/biotech \
  --host 127.0.0.1 \
  --port 8765
```

Open <http://127.0.0.1:8765>.

`--browse-root` controls which directories the browser-based file picker may
open. Repeat it to expose more than one location. On an HPC installation, for
example:

```bash
downstream-studio \
  --workspace /data/horse/ws/YOUR_PROJECT/downstream-studio-data \
  --browse-root /data/horse/ws/YOUR_PROJECT \
  --browse-root /data/cat/ws/YOUR_PROJECT
```

Only configured roots and their descendants are visible through the API.

## First project

1. Create a project.
2. Register the embeddings directory:

   ```text
   /Users/divar/Documents/biotech/af_random_100k_mixed
   ```

3. Register the target file:

   ```text
   /Users/divar/Documents/biotech/interface_af_features.csv
   ```

4. Open **New experiment**, choose models and fields, configure training, and
   select **Start training**. For an MLP, add or remove hidden layers and set
   each layer's neuron count and activation in the visual architecture editor.
5. The run page updates automatically and exposes metrics and artifacts when
   training finishes.

## Data locations

The `--workspace` directory contains:

```text
studio.sqlite3       project and experiment metadata
uploads/             browser-uploaded files
runs/<run-id>/       logs, exact config, command, metrics, and predictions
trash/               recoverable folders from deleted experiments
```

Do not place the workspace in the source checkout on a shared deployment. For
HPC usage, put it in a workspace directory with suitable quota and permissions.

## Architecture and scaling

The browser communicates with a small REST server. Training is executed in a
separate Python process, so a failed or memory-intensive run does not take down
the interface. `RunManager` is the executor boundary. A Slurm executor can be
added alongside the local executor to generate `sbatch` scripts, persist job
IDs, poll `squeue`/`sacct`, and collect the same output artifacts.

This initial release is intended for one trusted user on localhost or through
an SSH tunnel. It has no authentication and must not be exposed directly to the
public internet.

## Training performance

Accelerator runs keep normalized training tensors on the CUDA or MPS device and
shuffle/batch them there, avoiding repeated host-to-device transfers. Individual
test predictions are retained only for the final epoch checkpoint, which avoids
large redundant in-memory and CSV outputs. The dashboard defaults to batch size
4096 and 50–100 epochs because the interface experiments observed so far mostly
plateau in that range; both remain configurable.

## Barnard access through an SSH tunnel

After installing and starting Studio on a Barnard allocation or trusted login
environment, bind to localhost and forward the port from your Mac:

```bash
ssh -L 8765:127.0.0.1:8765 afza787g@login1.barnard.hpc.tu-dresden.de
```

Then open <http://127.0.0.1:8765> locally. For production HPC use, implement
the Slurm executor rather than performing training inside the web-server
process or on a login node.
