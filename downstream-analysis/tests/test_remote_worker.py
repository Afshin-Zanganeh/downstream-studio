from interface_downstream.remote_worker import _probe_command


def test_remote_worker_builds_argument_list_without_shell():
    config = {
        "embeddings": ["/data/model a/embeddings.csv"], "targets_csv": "/data/targets.csv",
        "output_dir": "/data/results/job", "target_columns": ["n_contacts"],
        "probes": ["linear"], "train_fractions": [0.5], "epochs": [25],
        "batch_size": 1024, "learning_rate": 0.001, "weight_decay": 0.0001,
        "hidden_dim": 128, "layers": [], "dropout": 0.1, "seed": 42, "device": "cpu",
    }
    command = _probe_command(config)
    assert "/data/model a/embeddings.csv" in command
    assert command[0] != "sh"
    assert "--targets-csv" in command
