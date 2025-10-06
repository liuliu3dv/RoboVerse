# 远程环境通讯

简洁的远程环境包装，让你可以像使用本地环境一样使用远程环境。

## 核心文件

- `remote_server.py` - 远程服务器，包装本地环境
- `remote_env.py` - 远程环境包装器，看起来像本地环境
- `simple_protocol.py` - 简化的通讯协议
- `example.py` - 完整使用示例

## 🚀 快速开始

### 方法1: 一键启动（推荐）

```bash
# 本地一键启动，自动连接远程服务器
python example.py --auto_start_remote --remote_host your_server_ip
```

### 方法2: 分步启动

#### 1. 启动服务端（远程机器）
```bash
# 方法1: 使用脚本
./start_server.sh

# 方法2: 直接运行
python example.py server
```

#### 2. 启动客户端（本地机器）
```bash
# 方法1: 使用脚本
./start_client.sh remote_host_ip 8888

# 方法2: 直接运行
python example.py
```

## 🔧 使用方法

### 服务端（远程机器）
```python
from remote_server import RemoteServer

# 创建你的本地环境
env = create_local_env("stack_cube")

# 创建服务器并设置环境
server = RemoteServer(port=8888)
server.set_environment(env)
server.start()
```

### 客户端（本地机器）
```python
from remote_env import RemoteEnv

# 创建远程环境
env = RemoteEnv("server_host", 8888)
env.connect()

# 像本地环境一样使用
obs = env.reset()
obs, reward, done, info = env.step(action)
state = env.get_state()
```

## 🌟 新功能：本地启动远程服务器

现在可以直接在本地启动远程服务器，无需手动在远程机器上操作：

### 使用SSH自动启动
```python
from remote_env import RemoteEnv

# 自动启动远程服务器
env = RemoteEnv("remote_host", 8888)
env.auto_start_remote_server(
    ssh_host="user@remote_host",
    remote_script_path="/path/to/RoboVerse/get_started/14_remote_isaac_local_mujoco",
    task_name="stack_cube"
)

# 连接并使用
env.connect()
obs = env.reset()
```

### 命令行一键启动
```bash
# 自动启动远程服务器并连接
python example.py --auto_start_remote --remote_host user@server_ip --task stack_cube
```

## 📋 环境配置

### 服务端环境要求
- 安装RoboVerse: `pip install -e .`
- 激活conda环境: `conda activate roboverse`
- 确保有GPU（如果使用IsaacSim）

### 客户端环境要求
- 只需要基本的Python环境
- 不需要GPU（除非本地也要渲染）

## 🐛 常见问题

### 1. 连接失败
```bash
# 检查服务端是否启动
netstat -tlnp | grep 8888

# 检查防火墙
sudo ufw status
```

### 2. 环境初始化失败
```bash
# 检查RoboVerse是否正确安装
python -c "import metasim; print('OK')"

# 检查任务是否存在
python -c "from metasim.task.registry import get_task_class; print(get_task_class('stack_cube'))"
```

### 3. 端口被占用
```bash
# 查看端口使用
lsof -i :8888

# 杀死占用进程
kill -9 <PID>
```

### 4. SSH自动启动失败
```bash
# 确保SSH密钥已配置
ssh-copy-id user@remote_host

# 测试SSH连接
ssh user@remote_host "ls"
```

## 🎯 使用场景

### 场景1: IsaacSim服务端 + MuJoCo客户端
```bash
# 远程机器（有GPU）- 自动启动
python example.py --auto_start_remote --remote_host gpu_server

# 本地机器
python example.py
```

### 场景2: MuJoCo服务端 + 无渲染客户端
```bash
# 远程机器
python example.py server

# 本地机器（只做控制，不渲染）
python -c "
from remote_env import RemoteEnv
env = RemoteEnv('remote_ip', 8888)
env.connect()
# 你的控制逻辑
"
```

### 场景3: 自定义环境
```python
# 服务端
class MyCustomEnv:
    def step(self, action):
        # 你的step逻辑
        return obs, reward, done, info
    
    def reset(self):
        # 你的reset逻辑
        return obs
    
    def get_state(self):
        # 你的get_state逻辑
        return state

server = RemoteServer()
server.set_environment(MyCustomEnv())
server.start()
```

## 🔧 高级用法

### 1. 多客户端连接
服务端天然支持多客户端，每个客户端都会收到相同的环境状态。

### 2. 自定义协议
可以扩展 `simple_protocol.py` 添加新的消息类型。

### 3. 错误处理
客户端会自动处理网络断开，并尝试重连。

### 4. 批量操作
```python
# 批量step操作
actions = [action1, action2, action3]
results = env.batch_step(actions)
```

## 📊 性能优化

- **网络延迟**: 使用本地网络或云服务器
- **批量操作**: 减少网络往返次数
- **状态缓存**: 避免重复获取相同状态
- **异步操作**: 并行处理多个请求

## 特点

- **简洁**: 只有4个核心文件
- **通用**: 可以包装任何有 `step()`, `reset()`, `get_state()` 方法的环境
- **透明**: 远程环境接口和本地环境完全一样
- **自动化**: 支持一键启动远程服务器