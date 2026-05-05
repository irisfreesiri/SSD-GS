import os
import subprocess
from pathlib import Path

# ================= GLOBAL CONFIG =================

PROJECT_ROOT = Path(__file__).resolve().parent

date = "20260430"
subtask = "real_nrhints"

data_root = PROJECT_ROOT / "data"
model_root = PROJECT_ROOT / "output"

dataset_name = "``Real_NRHints``"
scene_list = [
    "Cat",
    "CatSmall",
    "CupFabric",
    "Fish",
    "FurScene",
    "Pikachu",
    "Pixiu",
]

view_num = 2000

env = os.environ.copy()
env["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# env["CUDA_LAUNCH_BLOCKING"] = "1"
env["TORCH_CUDA_ARCH_LIST"] = "8.6"   # RTX 3090 = 8.6

test_iterations = [
    2000, 7000, 10000, 15000, 20000, 25000,
    30000, 40000, 50000, 60000, 70000,
    80000, 90000, 100000
]

save_iterations = [
    7000, 10000, 15000, 20000,
    30000, 40000, 50000, 60000,
    70000, 80000, 90000, 100000
]

checkpoint_iterations = [
    7000, 10000, 15000, 20000, 25000,
    30000, 40000, 50000, 60000,
    70000, 80000, 90000, 100000
]

render_iterations = [100000, 30000, 7000]
max_iteration = 100000

# ================= HELPER =================

def run_cmd(cmd, env=None):
    print("\n" + "=" * 80)
    print("Running command:")
    print(" ".join(map(str, cmd)))
    print("=" * 80 + "\n")

    subprocess.run(cmd, env=env, check=True)


# ================= MAIN LOOP =================

for scene_name in scene_list:
    print(f"\n\n########## Processing scene: {dataset_name}-{scene_name} ##########\n")

    source_path = data_root / dataset_name / scene_name
    model_path = model_root / f"{subtask}_{scene_name}_{date}"

    train_cmd = [
        "python", "train.py",
        "-s", str(source_path),
        "-m", str(model_path),

        "--data_device", "cpu",
        "--view_num", str(view_num),
        "--iterations", str(max_iteration),

        "--asg_freeze_step", "22000",
        "--specular_freeze_step", "9000",
        "--fit_linear_step", "7000",

        "--asg_lr_freeze_step", "40000",
        "--asg_lr_max_steps", "50000",
        "--asg_lr_init", "0.01",
        "--asg_lr_final", "0.0001",

        "--local_q_lr_freeze_step", "40000",
        "--local_q_lr_init", "0.01",
        "--local_q_lr_final", "0.0001",
        "--local_q_lr_max_steps", "50000",

        "--neural_phasefunc_lr_init", "0.001",
        "--neural_phasefunc_lr_final", "0.00001",
        "--freeze_phasefunc_steps", "50000",
        "--neural_phasefunc_lr_max_steps", "50000",

        "--position_lr_max_steps", "70000",
        "--densify_until_iter", "90000",
        "--unfreeze_iterations", "5000",

        "--test_iterations", *map(str, test_iterations),
        "--save_iterations", *map(str, save_iterations),
        "--checkpoint_iterations", *map(str, checkpoint_iterations),

        "--use_nerual_phasefunc",
        "--cam_opt",                # real-world only
        "--pl_opt",                 # real-world only
        "--eval",
    ]

    run_cmd(train_cmd, env=env)

    for iteration in render_iterations:
        render_cmd = [
            "python", "render.py",
            "-m", str(model_path),
            "--iteration", str(iteration),
            "--opt_pose",           # real-world only
            "--use_nerual_phasefunc",
        ]

        run_cmd(render_cmd, env=env)

    metrics_cmd = [
        "python", "metrics.py",
        "-m", str(model_path),
    ]

    run_cmd(metrics_cmd, env=env)