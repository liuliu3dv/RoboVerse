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


def replay_demo(hdf5_path, demo_index=0, render=True, speed=1.0, env=None):
    """Replay一个演示

    Args:
        hdf5_path: HDF5文件路径
        demo_index: 演示索引
        render: 是否渲染
        speed: 播放速度倍率
        env: 已创建的环境（如果为None则创建新环境）

    Returns:
        (success, total_steps, success_steps): 元组包含是否成功、总步数、成功步数
    """
    print(f"\n{'=' * 80}")
    print(f"Replay 演示 {demo_index}")
    print(f"{'=' * 80}")

    # 加载HDF5文件
    with h5py.File(hdf5_path, "r") as f:
        # 如果没有提供环境，创建新环境
        should_close_env = False
        if env is None:
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
                should_close_env = True
            except Exception as e:
                print(f"✗ 环境创建失败: {e}")
                import traceback

                traceback.print_exc()
                return None
            if render:
                env.render()
        # 获取演示数据
        demo_key = f"demo_{demo_index}"
        if demo_key not in f["data"]:
            available_demos = [k for k in f["data"].keys() if k.startswith("demo_")]
            print(f"\n✗ 演示 {demo_key} 不存在")
            print(f"可用演示: {available_demos[:10]}...")
            if should_close_env:
                env.close()
            return None

        demo = f["data"][demo_key]

        # 读取数据
        actions = demo["actions"][:]
        states = demo["states"][:]

        print(f"  步数: {len(actions)}")
        for body_name in env.sim.model.body_names:
            if body_name:  # 有些 body 没有名字
                pos = env.sim.data.get_body_xpos(body_name)
                print(f"Body {body_name:20s} -> {pos}")
        # Reset环境并设置初始状态
        env.reset()
        for body_name in env.sim.model.body_names:
            if body_name:  # 有些 body 没有名字
                pos = env.sim.data.get_body_xpos(body_name)
                print(f"Body {body_name:20s} -> {pos}")
        # 设置初始状态
        initial_state = states[0]
        try:
            env.sim.set_state_from_flattened(initial_state)
            env.sim.forward()
            print("\n-- Bodies --")
            for body_name in env.sim.model.body_names:
                if body_name:  # 有些 body 没有名字
                    pos = env.sim.data.get_body_xpos(body_name)
                    print(f"Body {body_name:20s} -> {pos}")
        except Exception as e:
            print(f"⚠ 警告: 无法设置初始状态: {e}")

        # 如果渲染，显示第一帧
        if render:
            env.render()
            env.render()
        # Replay每一步
        step_time = 1.0 / (20.0 * speed)  # 假设20Hz控制频率
        success_count = 0
        final_success = False

        for t in range(len(actions)):
            action = actions[t]

            # 执行动作
            obs, reward, done, info = env.step(action)

            # 检查成功
            if info.get("success", False):
                success_count += 1
                final_success = True

            # 渲染
            if render:
                env.render()
                time.sleep(step_time)

            # 如果完成，提前结束
            if done:
                break

        # 保持显示一段时间（仅当渲染且成功时）
        if render and final_success:
            time.sleep(0.5)

        # 关闭环境（仅当本函数创建时）
        if should_close_env:
            env.close()

        print(f"  结果: {'✓ 成功' if final_success else '✗ 失败'} (成功步数: {success_count}/{len(actions)})")

        return (final_success, len(actions), success_count)


def replay_all_demos(hdf5_path, max_demos=None, render=True, speed=1.0):
    """Replay所有演示并统计成功率

    Args:
        hdf5_path: HDF5文件路径
        max_demos: 最大演示数量（None表示全部）
        render: 是否渲染
        speed: 播放速度倍率
    """
    print(f"\n{'=' * 80}")
    print("批量Replay LIBERO演示")
    print(f"{'=' * 80}")
    print(f"\nHDF5文件: {hdf5_path}")

    # 获取演示数量
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
        demo_keys.sort(key=lambda x: int(x.split("_")[1]))

        if max_demos is not None:
            demo_keys = demo_keys[:max_demos]

        total_demos = len(demo_keys)
        print(f"总演示数: {total_demos}")
        print(f"渲染: {'是' if render else '否'}")
        print(f"播放速度: {speed}x")

    # 统计信息
    success_demos = 0
    failed_demos = 0
    results = []

    # 逐个replay
    for i, demo_key in enumerate(demo_keys):
        demo_index = int(demo_key.split("_")[1])

        result = replay_demo(
            hdf5_path=hdf5_path,
            demo_index=demo_index,
            render=render,
            speed=speed,
            env=None,  # 每个demo创建新环境
        )

        if result is not None:
            success, total_steps, success_steps = result
            results.append((demo_index, success, total_steps, success_steps))

            if success:
                success_demos += 1
            else:
                failed_demos += 1

    # 打印统计结果
    print(f"\n{'=' * 80}")
    print("统计结果")
    print(f"{'=' * 80}")
    print(f"\n总演示数: {total_demos}")
    print(f"成功: {success_demos} ({success_demos / total_demos * 100:.1f}%)")
    print(f"失败: {failed_demos} ({failed_demos / total_demos * 100:.1f}%)")

    # 打印详细结果
    print(f"\n详细结果:")
    for demo_index, success, total_steps, success_steps in results:
        status = "✓" if success else "✗"
        print(f"  Demo {demo_index}: {status} ({success_steps}/{total_steps} 步成功)")

    print(f"\n{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="Replay LIBERO演示数据")
    parser.add_argument(
        "--hdf5",
        type=str,
        default="roboverse_pack/tasks/libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5",
        help="HDF5文件路径",
    )
    parser.add_argument("--demo-index", type=int, default=None, help="演示索引 (默认: None，replay所有)")
    parser.add_argument("--max-demos", type=int, default=None, help="最大演示数量 (默认: None，全部)")
    parser.add_argument("--no-render", action="store_true", help="不渲染（无头模式）")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍率 (默认: 1.0)")

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.hdf5):
        print(f"✗ 错误: 文件不存在: {args.hdf5}")
        return

    # Replay演示
    if args.demo_index is not None:
        # Replay单个演示
        result = replay_demo(
            hdf5_path=args.hdf5, demo_index=args.demo_index, render=not args.no_render, speed=args.speed
        )
        if result:
            success, total_steps, success_steps = result
            print(f"\n最终结果: {'成功' if success else '失败'}")
    else:
        # Replay所有演示
        replay_all_demos(hdf5_path=args.hdf5, max_demos=args.max_demos, render=not args.no_render, speed=args.speed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback

        traceback.print_exc()
