"""Configuration values reported in the manuscript and revision response."""

STATIONS = [50953, 52203, 59097, 58238, 54623]

MODEL_CONFIGS = {
    "transformer": {
        "hidden_dim": 160,
        "nhead": 4,
        "num_transformer_layers": 3,
        "parameters": 943384,
        "jacobian_method": "vmap+jacrev",
        "jac_weight": 1e-5,
        "jac_freq": 1,
        "finetune_epochs": 100,
        "finetune_early_stop": 25,
    },
    "bilstm": {
        "hidden_dim": 180,
        "num_layers": 2,
        "parameters": 932474,
        "jacobian_method": "autograd.grad",
        "jac_weight": 5e-4,
        "jac_freq": 50,
        "finetune_epochs": 150,
        "finetune_early_stop": 30,
    },
    "mlp": {
        "hidden_dim": 12,
        "num_layers": 3,
        "parameters": 1189406,
        "jacobian_method": "vmap+jacrev",
        "jac_weight": 5e-4,
        "jac_freq": 1,
        "finetune_epochs": 150,
        "finetune_early_stop": 30,
    },
    "simple_lstm": {
        "hidden_dim": 128,
        "num_layers": 2,
        "parameters": 370070,
        "jacobian_method": "autograd.grad",
        "jac_weight": 0.0,
        "jac_freq": 1,
        "finetune_epochs": 0,
        "finetune_early_stop": 0,
    },
}

PRETRAIN_PROTOCOL = {
    "max_epochs": 1000,
    "early_stop_patience": 25,
    "train_batch_size": 256,
    "valid_batch_size": 1024,
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "ReduceLROnPlateau",
    "scheduler_factor": 0.5,
    "scheduler_patience": 5,
    "scheduler_min_lr": 1e-6,
}

BENCHMARK_PROTOCOL = {
    "cpu_threads": 8,
    "forward_batch_size": 5120,
    "jacobian_batch_size": 64,
    "host_to_device_transfer_timed": False,
}

__all__ = [
    "STATIONS",
    "MODEL_CONFIGS",
    "PRETRAIN_PROTOCOL",
    "BENCHMARK_PROTOCOL",
]
