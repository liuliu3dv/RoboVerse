# 远程环境通讯 (Remote Environment)

**核心思路**：远程运行真实任务，本地同步渲染

## 🎯 工作流程

```
┌──────────────────────┐         ┌──────────────────────┐
│  Remote (IsaacSim)   │         │  Local (MuJoCo)      │
├──────────────────────┤         ├──────────────────────┤
│ 1. Load trajectory   │         │ 1. Create MuJoCo env │
│ 2. Reset env         │────────>│ 2. Connect to remote │
│ 3. Step with action  │  state  │ 3. Set state         │
│ 4. Get state         │────────>│ 4. Render            │
│ 5. Repeat...         │         │ 5. Save video        │
└──────────────────────┘         └──────────────────────┘
```

**关键**：远程step一步 → 立即获取state → 本地set_state → 渲染

## 📦 核心文件

- `remote_env.py` (287行) - 远程环境客户端
- `remote_server.py` (145行) - 通用服务器
- `task_server.py` (136行) - 任务服务器（真实环境）
- `simple_protocol.py` (45行) - 通讯协议
- `remote_replay_sync.py` (240行) - 同步replay示例

**总计**: 853行核心代码

## 🚀 快速开始

```bash
# 运行同步replay（推荐）
python remote_replay_sync.py
```

这会：
1. 在远程服务器运行IsaacSim环境
2. 加载真实轨迹（traj文件）
3. 远程执行actions
4. 实时同步states到本地MuJoCo
5. 本地渲染并保存视频

## 💻 使用示例

### 基本用法

```python
from remote_env import RemoteEnv

# 创建远程环境（自动SSH隧道）
with RemoteEnv(
    remote_host="pabrtxl2.ist.berkeley.edu",
    ssh_host="rll_6000_2",
    use_tunnel=True,
    python_path="/path/to/python",
    remote_script_path="/path/to/scripts"
) as remote_env:
    # 启动远程服务器
    remote_env.start_remote_server(
        server_script="task_server.py",
        task_name="stack_cube",
        simulator="isaacsim"
    )
    
    # 设置隧道并连接
    remote_env.setup_tunnel()
    remote_env.connect()
    
    # 创建本地环境
    local_env = create_local_mujoco_env()
    
    # 同步replay
    obs = remote_env.reset()
    for action in trajectory_actions:
        # 远程执行
        obs, reward, done, info = remote_env.step(action)
        
        # 获取state并同步到本地
        state = remote_env.get_state()
        local_env.handler.set_states(state)
        local_env.handler.refresh_render()
```

## 🔧 配置

### SSH配置 (~/.ssh/config)

```
Host rll_6000_2
    HostName pabrtxl2.ist.berkeley.edu
    User ghr
    IdentityFile ~/.ssh/id_rsa
```

### 服务器配置

在 `remote_replay_sync.py` 中修改：

```python
config = {
    "remote_host": "pabrtxl2.ist.berkeley.edu",
    "ssh_host": "rll_6000_2",
    "python_path": "/datasets/v2p/current/murphy/dev/lab/bin/python",
    "remote_script_path": "/path/to/scripts",
    "task": "stack_cube",  # 任务名称
    "simulator": "isaacsim",  # 远程simulator
}
```

## 📊 性能

基于 rll_6000_2 测试：
- Reset: ~60ms
- Step: ~95ms
- Get State: ~20ms
- 100步任务: ~10秒

## 🎯 关键特性

### 1. 同步Replay
- 远程step → 立即本地set_state
- 实时渲染
- 无需等待收集完成

### 2. 真实轨迹
- 从traj文件加载真实actions
- 使用真实的任务环境
- 真实的reward和done

### 3. 自动化
- 自动SSH隧道
- 自动服务器管理
- 自动资源清理

### 4. 监控
- 日志监控
- 进程检查
- 错误处理

## 🐛 故障排查

### 连接失败
```bash
# 检查SSH
ssh rll_6000_2 "echo test"

# 检查端口
ssh rll_6000_2 "netstat -tlnp | grep 8888"
```

### 服务器启动失败
```bash
# 查看日志
ssh rll_6000_2 "cat /path/to/server.log"

# 检查进程
ssh rll_6000_2 "ps aux | grep task_server"
```

### 轨迹文件不存在
```bash
# 检查traj文件
ls roboverse_data/trajs/
```

## 📝 注意事项

1. **环境一致性**: 远程和本地的任务配置要一致
2. **轨迹文件**: 确保traj文件存在
3. **网络延迟**: 远程操作有~95ms延迟
4. **资源清理**: 使用context manager自动清理

## 🎉 优势

- **利用远程GPU**: 运行高精度IsaacSim
- **本地可视化**: MuJoCo低延迟渲染
- **真实轨迹**: 使用真实的demo数据
- **同步实时**: 边执行边渲染，无需等待

## 📚 文档

- `README.md` (本文件) - 快速开始
- `USAGE.md` - 详细API文档
- `FINAL_SUMMARY.md` - 项目总结
