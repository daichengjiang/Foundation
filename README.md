# CrazyE2E
export PYTHONPATH=$(pwd)
RL based ultra lightweight end-to-end navigation with Crazyflie

## Pre-requisites

```bash
git clone https://github.com/Jason-xy/CrazyE2E.git
cd CrazyE2E
./scripts/init.sh
```

## Simulator Scripts and Usage

### ./scripts/start.sh

If command arguments are provided, they will be passed to the container's entrypoint.

Example:

```bash
./scripts/start.sh /workspace/isaaclab/e2e_drone/rsl_rl/play.py --task v0 --checkpoint /workspace/isaaclab/e2e_drone/tasks/v0/model/model.pt --num_envs=16
```

If no arguments are provided, the container will use its default entrypoint (i.e. `bash`).

### ./e2e_drone/run.py

This is the main script that runs the end-to-end navigation. Default running script in `streaming` mode. If you want to run the `gui` mode, you can use `--gui` argument.

How to connect to the streaming ui:

1. Download the `Isaac Sim WebRTC Streaming Client` at https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html#isaac-sim-latest-release

2. Connect with the server ip.

Note: The `Windows` version of the client has a bug that causes the server to crash when disconnecting.

Usage (inside the container):

```bash
python run.py <script> <args>
```

Training example:

```bash
./scripts/start.sh /workspace/isaaclab/e2e_drone/rsl_rl/train.py --task v0
```

Playing example:

```bash
./scripts/start.sh /workspace/isaaclab/e2e_drone/rsl_rl/play.py --task v0 --checkpoint /workspace/isaaclab/e2e_drone/tasks/v0/model/model.pt --num_envs=16
```

### ./e2e_drone/tensorboard.sh

Usage (inside the container):

```bash
./tensorboard.sh <log_dir> # Default log_dir is /workspace/isaaclab/logs
```