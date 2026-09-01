import os

folders = [
    "data/raw/cicids2017",
    "data/raw/cicids2018",
    "data/raw/unsw_nb15",
    "data/raw/sample_pcaps",
    "data/interim/cleaned",
    "data/interim/packet_features",
    "data/interim/flow_features",
    "data/processed/windows",
    "data/processed/graphs",
    "data/processed/splits",
    "configs",
    "notebooks",
    "src/data",
    "src/features",
    "src/models",
    "src/training",
    "src/inference",
    "src/utils",
    "app/pages",
    "app/assets",
    "models/baselines",
    "models/gru",
    "models/gnn",
    "models/scalers",
    "outputs/figures",
    "outputs/reports",
    "outputs/metrics",
    "outputs/predictions",
    "tests",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

init_dirs = ["src", "src/data", "src/features", "src/models", "src/training", "src/inference", "src/utils"]
for d in init_dirs:
    init_path = os.path.join(d, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            pass

print("Folder structure created successfully!")
