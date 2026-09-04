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
- Target predictability ranking ordered by the best held-out R², with Pearson
  correlation and normalized error shown for the same best configuration.
- Full target-feature comparison charts for test R², Pearson correlation, and
  normalized MAE, using the best held-out configuration for each target.
- Experiment history with open, reuse-as-new, and recoverable delete actions.
- SQLite metadata and filesystem artifacts; no external service is required.

## Local installation

Install the analysis engine and Studio in the same Python environment:

```bash
cd /path/to/downstream-studio/downstream-analysis
python3 -m pip install -e '.[csv]'

cd /path/to/downstream-studio/downstream-studio
python3 -m pip install -e .
```

Start the application:

```bash
downstream-studio \
  --workspace /path/to/downstream-studio-data \
  --browse-root /path/to/data \
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
   /path/to/af_random_100k_mixed
   ```

3. Register the target file:

   ```text
   /path/to/interface_af_features.csv
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

The browser communicates with a small local REST server. `RunManager` selects a
local subprocess or the outbound SSH/Slurm executor. The latter uploads bounded
job descriptions, persists Slurm job IDs, polls `squeue`/`sacct`, and collects
the same result summaries without opening a web port on the HPC.

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

## Outbound SSH and Slurm execution

Studio can remain on the local workstation while datasets and computation stay
on TU Dresden HPC. No Studio web server or inbound port is opened there.

1. Create an ED25519 SSH key with a passphrase, install its public key on the
   HPC, and load the private key into the local SSH agent.
2. In **HPC setup**, select a Barnard or Capella login node, choose **SSH
   agent**, enter the ZIH username, and test the connection.
3. Set **Remote Python** to a Python 3.10+ executable available on login and
   compute nodes. Click **Deploy worker**. Studio creates:

   ```text
   ~/.downstream-studio/worker/
   ~/.downstream-studio/jobs/
   ~/.downstream-studio/results/
   ```

   Deployment uploads a versioned source archive, creates a private virtual
   environment, and installs `downstream-analysis[csv]`. The first deployment
   can take several minutes while PyTorch and pandas are installed remotely.
4. In **Data**, choose **HPC path** and register the embeddings directory and
   target CSV. Only schema information and bounded sample rows cross SSH.
5. In **Train**, choose **HPC via Slurm**, select the partition and resources,
   and submit the experiment.
6. Experiment pages monitor Slurm, stream the log, and copy result summaries to
   the local workspace. Predictions are copied only after completion and only
   below 100 MB; larger files remain on HPC.

Remote dataset transformations are intentionally disabled. Register original
remote files and select target fields in the experiment builder. Shell-sensitive
values are validated, remote directories use user-only permissions, and the
worker provides no arbitrary command endpoint.
