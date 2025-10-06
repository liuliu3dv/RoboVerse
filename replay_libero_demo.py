#!/usr/bin/env python3
"""使用LIBERO原版环境replay演示数据

这个脚本加载LIBERO的HDF5演示数据，并在仿真器中播放
"""

import argparse
import os
import sys
import time

import h5py

# 添加LIBERO路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "roboverse_data/LIBERO"))

try:
    from libero.libero import get_libero_path
    from libero.libero.envs import *

    print("✓ LIBERO导入成功")
except ImportError as e:
    print("✗ 错误: 无法导入LIBERO模块")
    print(f"  {e}")
    print("\n请确保:")
    print("  1. robosuite已安装: pip install robosuite")
    print("  2. LIBERO在正确路径: roboverse_data/LIBERO")
    sys.exit(1)


def replay_demo(hdf5_path, demo_index=0, render=True, speed=1.0):
    """Replay一个演示

    Args:
        hdf5_path: HDF5文件路径
        demo_index: 演示索引
        render: 是否渲染
        speed: 播放速度倍率
    """
    print(f"\n{'=' * 80}")
    print("Replay LIBERO演示")
    print(f"{'=' * 80}")
    print(f"\nHDF5文件: {hdf5_path}")
    print(f"演示索引: {demo_index}")
    print(f"播放速度: {speed}x")
    print(f"渲染: {'是' if render else '否'}")

    # 加载HDF5文件
    with h5py.File(hdf5_path, "r") as f:
        # 读取环境配置
        env_args_str = f["data"].attrs["env_args"]
        import json

        env_args = json.loads(env_args_str)

        env_name = env_args["env_name"]
        problem_name = env_args["problem_name"]
        env_kwargs = env_args["env_kwargs"]

        print("\n环境信息:")
        print(f"  名称: {env_name}")
        print(f"  任务: {problem_name}")
        print(f"  机器人: {env_kwargs['robots']}")

        # 修正BDDL文件路径（如果是旧路径）
        if "bddl_file_name" in env_kwargs:
            old_bddl = env_kwargs["bddl_file_name"]
            if "chiliocosm" in old_bddl or not os.path.exists(old_bddl):
                # 提取任务名
                task_name = os.path.basename(old_bddl)
                # 构建新路径
                new_bddl = f"roboverse_data/LIBERO/libero/libero/bddl_files/libero_90/{task_name}"
                if os.path.exists(new_bddl):
                    print("  修正BDDL路径:")
                    print(f"    旧: {old_bddl}")
                    print(f"    新: {new_bddl}")
                    env_kwargs["bddl_file_name"] = new_bddl
                else:
                    print(f"  ⚠ 警告: BDDL文件不存在: {new_bddl}")

        # 更新环境参数以支持渲染
        # 注意：如果use_camera_obs=True，必须has_offscreen_renderer=True
        use_camera = env_kwargs.get("use_camera_obs", False)

        if use_camera:
            # 有相机观测时，必须使用offscreen renderer
            env_kwargs["has_offscreen_renderer"] = True
            env_kwargs["has_renderer"] = render  # 可选的窗口显示
        else:
            # 没有相机观测时，可以只用renderer
            env_kwargs["has_renderer"] = render
            env_kwargs["has_offscreen_renderer"] = False

        if render or use_camera:
            if "camera_names" not in env_kwargs or not env_kwargs["camera_names"]:
                env_kwargs["camera_names"] = ["agentview"]
            if "camera_heights" not in env_kwargs:
                env_kwargs["camera_heights"] = 512
            if "camera_widths" not in env_kwargs:
                env_kwargs["camera_widths"] = 512

        # 创建环境
        print("\n创建环境...")
        try:
            env = TASK_MAPPING[problem_name](**env_kwargs)
            print("✓ 环境创建成功")
        except Exception as e:
            print(f"✗ 环境创建失败: {e}")
            import traceback

            traceback.print_exc()
            return
        env.render()
        env.render()
        # 获取演示数据
        demo_key = f"demo_{demo_index}"
        if demo_key not in f["data"]:
            available_demos = [k for k in f["data"].keys() if k.startswith("demo_")]
            print(f"\n✗ 演示 {demo_key} 不存在")
            print(f"可用演示: {available_demos[:10]}...")
            return

        demo = f["data"][demo_key]

        # 读取数据
        actions = demo["actions"][:]
        states = demo["states"][:]

        print("\n演示数据:")
        print(f"  步数: {len(actions)}")
        print(f"  动作维度: {actions.shape[1]}")
        print(f"  状态维度: {states.shape[1]}")
        if render:
            env.render()
            env.render()
        # Reset环境并设置初始状态
        print("\n开始replay...")
        env.reset()

        # 设置初始状态
        initial_state = states[0]
        try:
            env.sim.set_state_from_flattened(initial_state)
            env.sim.forward()
            print("✓ 初始状态设置成功")
        except Exception as e:
            print(f"⚠ 警告: 无法设置初始状态: {e}")

        # 如果渲染，显示第一帧
        if render:
            env.render()
            env.render()
        # Replay每一步
        print("\nReplay中...")
        step_time = 1.0 / (20.0 * speed)  # 假设20Hz控制频率
        env.render()
        success_count = 0
        for t in range(len(actions)):
            action = actions[t]

            # 执行动作
            obs, reward, done, info = env.step(action)

            # 检查成功
            if info.get("success", False):
                success_count += 1

            # 渲染
            if render:
                env.render()
                time.sleep(step_time)

            # 打印进度
            if (t + 1) % 50 == 0 or t == len(actions) - 1:
                progress = (t + 1) / len(actions) * 100
                print(
                    f"  进度: {t + 1}/{len(actions)} ({progress:.1f}%) - 奖励: {reward:.3f} - 成功: {success_count > 0}"
                )

            # 如果完成，提前结束
            if done:
                print(f"\n✓ 任务完成于步骤 {t + 1}")
                break

        # 保持显示一段时间
        if render:
            print("\n保持显示3秒...")
            time.sleep(3)

        # 关闭环境
        env.close()

        print(f"\n{'=' * 80}")
        print("Replay完成!")
        print(f"  总步数: {len(actions)}")
        print(f"  成功标记数: {success_count}")
        print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="Replay LIBERO演示数据")
    parser.add_argument(
        "--hdf5",
        type=str,
        default="roboverse_pack/tasks/libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5",
        help="HDF5文件路径",
    )
    parser.add_argument("--demo-index", type=int, default=0, help="演示索引 (默认: 0)")
    parser.add_argument("--no-render", action="store_true", help="不渲染（无头模式）")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍率 (默认: 1.0)")

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.hdf5):
        print(f"✗ 错误: 文件不存在: {args.hdf5}")
        return

    # Replay演示
    replay_demo(hdf5_path=args.hdf5, demo_index=args.demo_index, render=not args.no_render, speed=args.speed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback

        traceback.print_exc()
