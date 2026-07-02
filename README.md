# sentry_sim

MuJoCo-based sentry simulation workspace for ROS 2 Jazzy.

## Workspace Layout

```text
sentry_sim/
├── src/
│   ├── sentry_sim_assets/
│   ├── sentry_sim_description/
│   ├── sentry_sim_bringup/
│   ├── sentry_sim_interfaces/
│   ├── sentry_sim_nav_plugins/
│   └── external/
│       └── RM2026-sentry/
├── docs/
├── scripts/
├── build/
├── install/
└── log/
```

## Package Roles

- `sentry_sim_assets`: MuJoCo scene assets, meshes, and imported field files.
- `sentry_sim_description`: Robot URDF/xacro and `robot_state_publisher` launch files.
- `sentry_sim_bringup`: MuJoCo node, simulation launch files, and runtime wiring.
- `sentry_sim_interfaces`: Reserved for future custom messages and services.
- `sentry_sim_nav_plugins`: Reserved for simulation-specific nav adapters.
- `src/external/RM2026-sentry`: External real-robot/navigation stack snapshot, prepared for later submodule conversion.

## Build

```bash
source /opt/ros/jazzy/setup.bash
cd /home/somo/dev/sentry_sim
colcon build --symlink-install
source install/setup.bash
```

## Run Simulation

```bash
source /opt/ros/jazzy/setup.bash
cd /home/somo/dev/sentry_sim
source install/setup.bash
ros2 launch sentry_sim_bringup sim.launch.py
```

If running from a non-graphical TTY, provide a display server before starting the viewer.

## Notes

- The repository root is now the only ROS 2 workspace root.
- Do not use the removed `sentry_sim_ws` layout anymore.
- Assets are resolved through `sentry_sim_assets`, not by repository-relative fallback paths alone.
- External domains under `src/external/RM2026-sentry/src` are disabled by default with domain-level `COLCON_IGNORE` files.
- See [docs/submodule-setup.md](file:///home/somo/dev/sentry_sim/docs/submodule-setup.md) for the git submodule conversion steps.
