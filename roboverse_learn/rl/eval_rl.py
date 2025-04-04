from __future__ import annotations

import os
import time

import imageio
import numpy as np
import rootutils
import wandb
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import hydra
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint


def set_np_formatting():
    """Formats numpy print"""
    np.set_printoptions(
        edgeitems=30,
        infstr="inf",
        linewidth=4000,
        nanstr="nan",
        precision=2,
        suppress=False,
        threshold=10000,
        formatter=None,
    )


@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    if cfg.environment.sim_name == "isaacgym":
        from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401
    import torch

    set_np_formatting()

    from metasim.cfg.scenario import ScenarioCfg
    from metasim.cfg.sensors import PinholeCameraCfg
    from metasim.utils.setup_util import SimType, get_robot, get_sim_env_class, get_task
    from roboverse_learn.rl.algos import get_algorithm

    # Override headless setting for evaluation
    cfg.environment.headless = False

    # Setup devices
    if cfg.experiment.multi_gpu:
        rank = int(os.getenv("LOCAL_RANK", "0"))
        sim_device = f"cuda:{rank}"
        rl_device = f"cuda:{rank}"
    else:
        sim_device = f"cuda:{cfg.experiment.device_id}" if cfg.experiment.device_id >= 0 else "cpu"
        rl_device = f"cuda:{cfg.experiment.device_id}" if cfg.experiment.device_id >= 0 else "cpu"

    # Environment setup
    cprint("Setting up environment for evaluation", "green", attrs=["bold"])
    task = get_task(cfg.train.task_name)
    robot = get_robot(cfg.train.robot_name)
    scenario = ScenarioCfg(task=task, robot=robot)
    scenario.cameras.append(
        PinholeCameraCfg(
            name="camera",
            data_types=["rgb", "depth"],
            width=64,
            height=64,
            pos=(1.5, 0.0, 1.5),
            look_at=(0.0, 0.0, 0.0),
        )
    )

    # Initialize environment
    tic = time.time()
    sim_name = cfg.environment.sim_name.lower()
    if sim_name == "isaacgym":
        from roboverse_learn.rl.envs.isaacgym_wrapper import IsaacGymWrapper

        env = IsaacGymWrapper(scenario, cfg.environment.num_envs, headless=False, seed=cfg.experiment.seed)
        env.launch()
    elif sim_name == "mujoco":
        from roboverse_learn.rl.envs.mujoco_wrapper import MujocoWrapper

        env = MujocoWrapper(scenario, cfg.environment.num_envs, headless=True, seed=cfg.experiment.seed)
        env.launch()
    elif sim_name == "isaaclab":
        from roboverse_learn.rl.envs.isaaclab_wrapper import IsaacLabWrapper

        env = IsaacLabWrapper(scenario, cfg.environment.num_envs, headless=False, seed=cfg.experiment.seed)
        env.launch()
    else:
        env_class = get_sim_env_class(SimType(cfg.environment.sim_name))
        env = env_class(scenario, cfg.environment.num_envs, headless=False)
        if hasattr(env, "set_seed"):
            env.set_seed(cfg.experiment.seed)

    if hasattr(env, "set_verbose"):
        env.set_verbose(False)
    toc = time.time()
    log.info(f"Time to launch environment: {toc - tic:.2f}s")

    # Output directory setup
    checkpoint_path = cfg.experiment.checkpoint
    if checkpoint_path is None:
        log.error("No checkpoint specified for evaluation. Use --experiment.checkpoint to specify.")
        return

    if not os.path.exists(checkpoint_path):
        log.error(f"Checkpoint not found: {checkpoint_path}")
        return

    output_dir = os.path.join("eval_outputs", cfg.experiment.output_name)
    os.makedirs(output_dir, exist_ok=True)
    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # Initialize wandb for evaluation logging
    wandb_run = None
    if cfg.experiment.get("use_wandb", True):
        wandb_project = cfg.experiment.get("wandb_project", "roboverse-eval")
        wandb_group = cfg.experiment.get("wandb_group", None)
        wandb_name = f"eval-{cfg.train.task_name}-{cfg.train.algo}-{cfg.experiment.output_name}"

        # Only initialize if wandb hasn't been initialized already
        if not wandb.run:
            try:
                wandb_run = wandb.init(
                    project=wandb_project,
                    group=wandb_group,
                    name=wandb_name,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    reinit=True,
                    tags=["evaluation"],
                )
                log.info(f"Initialized wandb run: {wandb_name}")
            except Exception as e:
                log.error(f"Failed to initialize wandb: {e}")
                wandb_run = None
        else:
            wandb_run = wandb.run
            log.info(f"Using existing wandb run: {wandb.run.name}")

    # Initialize agent
    algo_name = cfg.train.algo.lower()
    agent = get_algorithm(algo_name, env=env, output_dif=output_dir, full_config=OmegaConf.to_container(cfg))
    log.info(f"Loading policy from checkpoint: {checkpoint_path}")
    agent.restore_test(checkpoint_path)

    # Evaluation settings
    num_eval_episodes = cfg.experiment.get("num_eval_episodes", 10)
    max_episode_steps = task.episode_length if hasattr(task, "episode_length") else 500
    save_video = cfg.experiment.get("save_video", True)

    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    success_rate = 0
    step_counter = 0  # Global step counter for wandb logging

    # Run evaluation episodes
    for episode in range(num_eval_episodes):
        log.info(f"Starting evaluation episode {episode + 1}/{num_eval_episodes}")

        # Reset environment
        obs_dict = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_success = False
        video_frames = [] if save_video else None

        # If environment returns a frame, save it
        if hasattr(env, "render"):
            frame = env.render()
            if frame is not None and save_video:
                video_frames.append(frame)

        # Episode loop
        done = torch.zeros(env.num_envs, dtype=torch.bool)
        while not done.all() and episode_steps < max_episode_steps:
            # Process observation through policy
            if hasattr(agent, "running_mean_std"):
                obs_normalized = agent.running_mean_std(obs_dict["obs"])
                input_dict = {"obs": obs_normalized}
                action = agent.model.act_inference(input_dict)
                action = torch.clamp(action, -1.0, 1.0)
            else:
                # Fallback for agents that don't use running_mean_std (dreamer)
                action = agent.predict(obs_dict["rgb"])

            # Step environment
            obs_dict, reward, done, timeout, info = env.step(action)
            episode_reward += reward.mean().item() if isinstance(reward, torch.Tensor) else reward
            episode_steps += 1
            step_counter += 1

            # Render and save frame if available
            if hasattr(env, "render") and save_video:
                frame = env.render()
                if frame is not None:
                    video_frames.append(frame)

            # Log step information
            if episode_steps % 10 == 0:
                log.info(
                    f"Episode {episode + 1}, Step {episode_steps}, Reward: {reward.mean().item() if isinstance(reward, torch.Tensor) else reward}"
                )

                # Log step metrics to wandb
                if wandb_run:
                    wandb.log({
                        "eval/step_reward": reward.mean().item() if isinstance(reward, torch.Tensor) else reward,
                        "eval/episode": episode + 1,
                        "eval/step": episode_steps,
                        "eval/global_step": step_counter,
                    })

        # Record episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        if episode_success:
            success_rate += 1

        # Save video
        if save_video and video_frames:
            video_path = os.path.join(video_dir, f"episode_{episode + 1}.mp4")
            imageio.mimsave(video_path, video_frames, fps=30)
            log.info(f"Saved video to {video_path}")

            # Log video to wandb
            if wandb_run:
                try:
                    wandb.log({
                        f"eval/episode_{episode + 1}_video": wandb.Video(
                            video_path,
                            fps=30,
                            format="mp4",
                            caption=f"Episode {episode + 1} - Reward: {episode_reward:.2f}",
                        ),
                        "eval/episode_reward": episode_reward,
                        "eval/episode_length": episode_steps,
                        "eval/episode_success": 1 if episode_success else 0,
                        "eval/episode": episode + 1,
                    })
                except Exception as e:
                    log.error(f"Failed to log video to wandb: {e}")

        log.info(
            f"Episode {episode + 1} finished: Steps={episode_steps}, Reward={episode_reward}, Success={episode_success}"
        )

    # Report overall results
    success_rate = success_rate / num_eval_episodes
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    log.info("=" * 50)
    log.info("Evaluation Results:")
    log.info(f"Success Rate: {success_rate:.2f}")
    log.info(f"Mean Episode Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    log.info(f"Mean Episode Length: {mean_length:.2f}")
    log.info("=" * 50)

    # Save results to file
    results = {
        "checkpoint": checkpoint_path,
        "task": cfg.train.task_name,
        "robot": cfg.train.robot_name,
        "algorithm": cfg.train.algo,
        "num_episodes": num_eval_episodes,
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }

    results_path = os.path.join(output_dir, "eval_results.pkl")
    import pickle

    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    log.info(f"Saved evaluation results to {results_path}")

    # Log final metrics to wandb
    if wandb_run:
        wandb.log({
            "eval/final/success_rate": success_rate,
            "eval/final/mean_reward": mean_reward,
            "eval/final/std_reward": std_reward,
            "eval/final/mean_length": mean_length,
        })

        # World model prediction video generation for Dreamer
        if (
            algo_name in ["dreamer", "dreamerv2", "dreamerv3"]
            and hasattr(agent, "world_model")
            and hasattr(agent.world_model, "video_pred")
        ):
            try:
                log.info("Generating world model prediction video...")
                # Sample data from agent's storage/buffer
                if hasattr(agent, "storage"):
                    # Check which method is available and use it
                    if hasattr(agent.storage, "sample_sequence"):
                        eval_batch = agent.storage.sample_sequence()
                    elif hasattr(agent.storage, "prepare_training"):
                        eval_batch = agent.storage.prepare_training()
                    else:
                        # Fallback to prepare_batch_data if available
                        eval_batch = agent.storage.prepare_batch_data()

                    # Generate world model predictions
                    with torch.no_grad():
                        video_pred = agent.world_model.video_pred(eval_batch)

                    # Convert to numpy
                    video_pred_np = video_pred.detach().cpu().numpy()

                    # Save to file
                    wm_video_path = os.path.join(video_dir, "world_model_pred.mp4")

                    # Process into a format that can be saved as video
                    # First normalize to [0, 255] range
                    video_frames = (video_pred_np * 255).astype(np.uint8)

                    # Reshape if needed to be compatible with imageio
                    # We're assuming video_pred has shape [batch, time, height, width, channels]
                    # or similar that can be processed
                    if len(video_frames.shape) == 5:  # [batch, time, height, width, channels]
                        frames_list = []
                        for t in range(video_frames.shape[1]):
                            # Concatenate all batches vertically
                            frame = np.concatenate([video_frames[b, t] for b in range(video_frames.shape[0])], axis=0)
                            frames_list.append(frame)
                        imageio.mimsave(wm_video_path, frames_list, fps=5)
                    else:
                        # Fallback for other formats
                        imageio.mimsave(wm_video_path, [video_frames], fps=5)

                    log.info(f"Saved world model prediction video to {wm_video_path}")

                    # Log to wandb
                    wandb.log({
                        "eval/world_model_predictions": wandb.Video(
                            wm_video_path, fps=5, format="mp4", caption="World Model Predictions"
                        )
                    })
            except Exception as e:
                log.error(f"Failed to generate world model prediction video: {e}")

        # Finish wandb run
        if wandb_run and wandb_run.id == wandb.run.id:  # Only finish if we created this run
            wandb.finish()


if __name__ == "__main__":
    main()
