# Interface Downstream Suite

A collaborative toolkit for measuring which protein-interface properties are encoded in learned embeddings.

The repository contains two cooperating packages:

- `downstream-analysis`: a reusable PyTorch package for linear and MLP regression probes.
- `downstream-studio`: a React and Python interface for managing datasets, configuring probes, running experiments, inspecting metrics, and connecting to HPC resources.

Large embeddings, exported database tables, experiment outputs, local workspace databases, Python environments, Node dependencies, and credentials are intentionally excluded from version control.

## Development

Install the analysis package in a virtual environment, then install the Studio backend and frontend dependencies. See each package's README for detailed commands.

Run the Studio backend from the repository root:

```bash
downstream-studio --workspace /path/to/local/workspace --browse-root /path/to/data --host 127.0.0.1 --port 8765
```

Run the React development server in a second terminal:

```bash
cd downstream-studio/frontend
npm install
npm run dev -- --host 127.0.0.1
```
