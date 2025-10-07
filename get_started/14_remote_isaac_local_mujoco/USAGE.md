# Remote Environment 使用指南

## 📦 核心文件

- `remote_env_v2.py` - 完整的远程环境类
- `remote_server.py` - 远程服务器
- `simple_protocol.py` - 通讯协议
- `remote_replay_example.py` - 使用示例

## 🚀 快速开始

### 1. 基本使用

```python
from remote_env_v2 import RemoteEnv

# 创建远程环境
with RemoteEnv(
    remote_host="server.example.com",
    port=8888,
    ssh_host="my_server",  # SSH config中的名称
    use_tunnel=True,
    python_path="/path/to/python",
    remote_script_path="/path/to/scripts"
) as env:
    # 启动远程服务器
    env.start_remote_server(cleanup_old=True)
    
    # 设置SSH隧道
    env.setup_tunnel()
    
    # 连接
    env.connect()
    
    # 使用环境
    obs = env.reset()
    obs, reward, done, info = env.step(action)
```

### 2. 运行示例

```bash
# 运行远程replay示例
python remote_replay_example.py
```

## 🔧 配置说明

### RemoteEnv参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `remote_host` | 远程服务器地址 | 必填 |
| `port` | 端口号 | 8888 |
| `ssh_host` | SSH配置名称 | remote_host |
| `use_tunnel` | 使用SSH隧道 | True |
| `python_path` | 远程Python路径 | "python" |
| `remote_script_path` | 远程脚本路径 | None |

### SSH配置示例

在 `~/.ssh/config` 中添加：

```
Host rll_6000_2
    HostName pabrtxl2.ist.berkeley.edu
    User ghr
    IdentityFile ~/.ssh/id_rsa
```

## 📋 功能特性

### 1. 自动SSH隧道
- 自动建立SSH隧道绕过防火墙
- 支持本地端口转发

### 2. 远程服务器管理
- 自动启动远程服务器
- 自动清理旧进程
- 进程状态监控

### 3. 日志监控
- 实时查看远程日志
- 错误检测和报告

### 4. 错误处理
- 连接超时处理
- 自动重连机制
- 优雅的资源清理

## 🎯 高级用法

### 1. 自定义服务器脚本

```python
env.start_remote_server(
    server_script="my_custom_server.py",
    task_name="pick_cube",
    num_envs=4,
    cleanup_old=True
)
```

### 2. 监控远程状态

```python
# 检查进程
process_info = env.check_remote_process("server_name")
if process_info:
    print(f"Server is running: {process_info}")

# 检查日志
log = env.check_remote_log("/path/to/server.log", lines=20)
if log:
    print(f"Server log:\n{log}")
```

### 3. Context Manager

```python
# 自动管理资源
with RemoteEnv(...) as env:
    env.start_remote_server()
    env.setup_tunnel()
    env.connect()
    # 使用环境...
# 自动清理资源
```

## 🔍 故障排查

### 1. 连接失败

**问题**: `Failed to connect: timed out`

**解决**:
- 检查SSH配置是否正确
- 确认远程服务器已启动
- 使用SSH隧道: `use_tunnel=True`

### 2. 服务器启动失败

**问题**: 日志显示错误

**解决**:
- 检查Python路径是否正确
- 确认远程脚本路径存在
- 查看完整日志: `env.check_remote_log(...)`

### 3. 端口被占用

**问题**: `Address already in use`

**解决**:
- 启用自动清理: `cleanup_old=True`
- 手动清理: `ssh host "pkill -f server.py"`

## 📊 性能指标

基于rll_6000_2测试结果：

- **Reset延迟**: ~60ms
- **Step延迟**: ~95ms
- **网络开销**: 通过SSH隧道约增加5-10ms

## 🎓 完整示例

参考 `remote_replay_example.py` 查看完整的使用示例，包括：
- 远程服务器启动
- SSH隧道设置
- 环境连接
- 任务执行
- 状态监控
- 资源清理

## 📝 注意事项

1. **SSH密钥**: 确保已配置SSH密钥免密登录
2. **网络延迟**: 远程环境会有网络延迟，适合训练但不适合实时控制
3. **资源清理**: 使用context manager或手动调用`close()`
4. **日志监控**: 定期检查远程日志确保服务器正常运行
