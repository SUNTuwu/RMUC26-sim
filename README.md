# sentry_sim

基于 ROS 2 Jazzy 和 MuJoCo 的哨兵仿真仓库。

当前仓库只维护仿真本体、传感器模拟、启动编排和与外部导航栈的对接；导航、定位、状态估计等能力来自 `src/external/RM2026-sentry-ws`。

## 仓库概览

```text
sentry_sim/
├── src/
│   ├── sim_assets/                # MuJoCo 场地与传感器网格资产
│   ├── sim_core/                  # 仿真核心节点、雷达桥接、控制适配
│   ├── sim_bringup/               # launch/config/rviz/urdf 编排入口
│   └── external/
│       └── RM2026-sentry-ws/      # 外部导航与业务工作区
├── scripts/                       # 统一的构建、启动、资产转换脚本
├── docs/                          # 预留文档目录
├── build/ install/ log/           # colcon 产物
└── .venv/                         # MuJoCo 等仿真侧 Python 环境
```

## 主要作用

- `sim_assets`
  - 安装 MuJoCo 所需的 `meshes/` 资产，供仿真场景和碰撞模型使用。
- `sim_core`
  - 提供 `sentry_sim_node`，负责 MuJoCo 仿真本体、场景装配、物理推进和传感器模拟。
  - 内部按 `runtime / components / adapters` 分层，分别承载仿真内核、组件执行和外部接口适配。
  - 提供 `components/_livox_bridge.py`，把仿真雷达点云编码成 `livox_ros_driver2/msg/CustomMsg` / PointCloud2，直接对接实机算法链。
  - 提供 `adapters/chassis_adapter.py`、`adapters/imu_adapter.py`、`keyboard_test`、`adapters/nav_feedback_adapter.py` 等控制、传感器与导航适配节点。
- `sim_bringup`
  - 提供 `sim.launch.py`，用于纯仿真启动。
  - 提供 `sim3d_nav.launch.py`，用于仿真接入外部状态估计、建图、导航链。
  - 保存运行参数 `config/sim_config.yaml`、可视化配置 `rviz/`、机器人描述 `urdf/sentry.urdf.xacro`。
- `src/external/RM2026-sentry-ws`
  - 保存外部导航/定位/状态估计/业务包。
  - 当前仓库通过脚本选择性构建其中的导航依赖，并在启动时将其 overlay 进来。

## 架构边界

- `sentry_sim_node` 维护内部真值，但不对外发布导航语义的 `/Odometry` 或 `odom -> base_link`。
- `odom -> base_link` 由状态估计节点负责，当前对接链路默认是 `point_lio`。
- `map -> odom` 由定位节点负责，例如 `dll_localization`。
- 仿真雷达直接发布 `livox_ros_driver2/msg/CustomMsg`，避免额外 bridge 进程和消息格式偏差。

### sim_core 内部分层

- `runtime`
  - 对应 `sim_core/runtime.py` 及其依赖的 `frame_tree.py`、`scene_builder.py`。
  - 负责加载机器人 frame 树、拼装 MuJoCo 场景、维护物理步进、仿真时钟和内部真值读取接口。
  - 只关心仿真内部状态与物理执行，不直接承接键盘、导航、串口这类外部协议语义。
- `components`
  - 对应 `sim_core/components/` 与 `component_manager.py`。
  - 由 `ComponentManager` 挂载到 `runtime` 周围，消费统一的内部控制入口 `/sim/cmd_vel`，并发布仿真可观测数据，例如 `/joint_states`、Livox 点云和原始 IMU。
  - `comp_chassis.py`、`comp_gimbal.py`、`comp_livox.py` 属于这一层；`_livox_bridge.py` 是 `comp_livox.py` 的私有编码辅助，不单独承担节点生命周期。
- `adapters`
  - 对应 `sim_core/adapters/` 下的独立 ROS 节点。
  - 负责把外部控制链或导航链的话题语义整理成仿真内部统一接口，或把仿真/估计结果转换成外部更需要的反馈接口。
  - `adapters/chassis_adapter.py` 负责把键盘控制整理成 `/sim/cmd_vel`；`adapters/imu_adapter.py` 负责把 `/sim/imu_*` 低通滤波后发布为 `/livox/imu_*`；`adapters/nav_feedback_adapter.py` 负责把 `/gimbal_Odometry`、`/joint_states` 等整理成 `/Odometry` 和外部反馈话题。

### 分层约束

- `runtime` 不直接依赖外部导航链的话题协议，也不吸收控制适配逻辑。
- `components` 只实现内部执行与传感器发布，不直接理解键盘模式、导航反馈协议或串口业务语义。
- `adapters` 不直接操作 MuJoCo 内部对象，只通过 ROS 话题与 `runtime/components` 交互。
- 新增功能时，若需求是“怎么模拟/怎么发布仿真观测”，优先放 `components`；若需求是“怎么把外部语义转进来或转出去”，优先放 `adapters`；若需求是“怎么建场景/推进物理/读内部真值”，放 `runtime`。

## scripts 目录

### 构建脚本

- `scripts/ninja_make_nav_deps.sh`
  - 在 `src/external/RM2026-sentry-ws` 中用 Ninja 构建外部导航依赖。
  - 当前会构建 `livox_ros_driver2`、`pointcloud_preprocessor`、`io_bringup`、`mapping_bringup`、`nav_bringup`、`main_bringup` 等包。
- `scripts/ninja_make_sim.sh`
  - 在仓库根目录构建仿真侧包：`sim_assets`、`sim_core`、`sim_bringup`。
  - 构建前会清理根工作区里与外部包重名的残留 overlay，避免混装。
  - 通过 `-DCMAKE_CXX_FLAGS=-DROS_${ROS_DISTRO^^}` 注入 ROS 发行版宏定义。

### 启动脚本

- `scripts/start_sim3d_mujoco.sh`
  - 激活 `.venv`，叠加外部工作区与本仓库的 install，然后启动 `ros2 launch sim_bringup sim.launch.py`。
  - 适合只看 MuJoCo 仿真、底盘控制、`robot_state_publisher` 和传感器输出。
- `scripts/start_sim3d_nav.sh`
  - 激活 `.venv`，叠加外部工作区与本仓库的 install，然后启动 `ros2 launch sim_bringup sim3d_nav.launch.py`。
  - 默认参数包括 `robot_type=sim_sentry_fold`、`lio=pointlio`，并关闭地图/定位文件输入。

### 工具脚本

- `scripts/usdc_to_mujoco.py`
  - 将 USD/USDC 场景导出为 OBJ 网格，并生成 MuJoCo 可引用的 XML 片段。
  - 适合把外部建模资产转换成 `sim_assets` 可消费的静态场景资源。

## 当前启动链路

### 1. 纯仿真

`sim_bringup/sim.launch.py` 会启动：

- `sim_core/chassis_adapter`
- `sim_core/imu_adapter`
- `robot_state_publisher`
- `sim_core/sentry_sim_node`

其中机器人模型来自 `sim_bringup/urdf/sentry.urdf.xacro`。

### 2. 仿真 + 导航

`sim_bringup/sim3d_nav.launch.py` 会在仿真之外继续接入：

- `pointcloud_preprocessor`
- `state_estimation_bringup/state_estimation.launch.py`
- `mapping_bringup/mapping.launch.py`
- `nav_bringup/nav.launch.py`
- `sim_core/nav_feedback_adapter`
- `nav_serial_driver_ch343/nav_serial_plugin_node`
- 可选 `rviz2`

仿真雷达话题来自 `sim_config.yaml` 中配置的 `left_lidar_ip` 和 `right_lidar_ip`，会被拼成：

- `/livox/lidar_<left_ip>/pointcloud`
- `/livox/lidar_<right_ip>/pointcloud`

## 关键配置

- `src/sim_bringup/config/sim_config.yaml`
  - `robot_init_location`: 初始位姿
  - `left_lidar_ip` / `right_lidar_ip`: 两个 Livox 话题命名来源
  - `boundary_x_min/max`、`boundary_y_min/max`: 场地边界
  - `max_tilt_deg`: 倾倒/姿态保护阈值
- `src/sim_bringup/rviz/sim3d_visualization.rviz`
  - 仿真导航联调时使用的 RViz 配置。
- `src/sim_bringup/urdf/sentry.urdf.xacro`
  - 底盘、云台和双雷达坐标系定义。

## 推荐构建流程

先构建外部导航依赖，再构建仿真包：

```bash
source /opt/ros/jazzy/setup.bash
cd /root/sentry_sim

./scripts/ninja_make_nav_deps.sh
./scripts/ninja_make_sim.sh
```

如果只修改仿真侧代码，通常只需重新执行：

```bash
cd /root/sentry_sim
./scripts/ninja_make_sim.sh
```

## 运行方式

### 启动纯仿真

```bash
cd /root/sentry_sim
./scripts/start_sim3d_mujoco.sh
```

可选关闭 MuJoCo Viewer：

```bash
cd /root/sentry_sim
ENABLE_VIEWER=false ./scripts/start_sim3d_mujoco.sh
```

### 启动仿真 + 导航

```bash
cd /root/sentry_sim
./scripts/start_sim3d_nav.sh
```

可通过环境变量切换机器人类型或 RViz：

```bash
cd /root/sentry_sim
ROBOT_TYPE=sim_sentry_fold USE_NAV_RVIZ=false ./scripts/start_sim3d_nav.sh
```

## 环境说明

- ROS 发行版默认读取 `ROS_DISTRO`，未设置时脚本默认 `jazzy`。
- `.venv` 主要承载 MuJoCo 等仿真专用 Python 包。
- 外部工作区 install 路径默认是 `src/external/RM2026-sentry-ws/install`。
- 非图形环境运行 Viewer 时，需要自行提供显示服务。

## 备注

- 根目录就是唯一的仿真工作区，不再使用旧的多层 workspace 布局。
- 外部工作区中很多 domain 默认可能通过 `COLCON_IGNORE` 控制是否参与构建，详见 `src/external/README.md`。
