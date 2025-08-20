"""Example usage of RslRlWrapperComplete with OnPolicyRunner."""

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl_wrapper_complete import RslRlWrapperComplete

# Import your scenario configuration
from metasim.scenario.scenario import ScenarioCfg
from metasim.constants import SimType


def create_training_config():
    """Create a training configuration for PPO."""
    return {
        "algorithm": {
            "class_name": "PPO",
            "learning_rate": 1e-3,
            "entropy_coef": 0.01,
            "learning_rate_schedule": "adaptive",
            "clip_range": 0.2,
            "clip_range_schedule": "linear",
            "value_loss_coef": 1.0,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "value_bootstrap": True,
            "gamma": 0.99,
            "lam": 0.95,
        },
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "actor_obs_branch": None,
            "critic_obs_branch": None,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "num_steps_per_env": 24,
            "max_iterations": 1000,
            "save_interval": 50,
            "empirical_normalization": True,
        },
        "num_steps_per_env": 24,
        "save_interval": 50,
        "empirical_normalization": True,
    }


def main():
    """Main training function."""

    # 1. Create your scenario configuration
    # Replace this with your actual scenario configuration
    scenario = ScenarioCfg(
        simulator=SimType.ISAACGYM,
        num_envs=4096,
        robots=[...],  # Your robot configuration
        task=...,      # Your task configuration
        # ... other configuration parameters
    )

    # 2. Create the wrapped environment
    env = RslRlWrapperComplete(scenario)

    # 3. Create training configuration
    train_cfg = create_training_config()

    # 4. Create OnPolicyRunner
    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir="./logs/ppo_training",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 5. Start training
    runner.learn(num_learning_iterations=1000)

    # 6. Save the final model
    runner.save("./models/final_model.pt")

    # 7. Close the environment
    env.close()


def test_wrapper():
    """Test the wrapper functionality."""

    # Create a simple test scenario (you'll need to adapt this to your actual scenario)
    scenario = ScenarioCfg(
        simulator=SimType.ISAACGYM,
        num_envs=4,  # Small number for testing
        robots=[...],  # Your robot configuration
        task=...,      # Your task configuration
    )

    # Create the wrapper
    env = RslRlWrapperComplete(scenario)

    # Test basic functionality
    print("Environment info:", env.get_env_info())

    # Test observation retrieval
    obs, extra_obs = env.get_observations()
    print(f"Observation shape: {obs.shape}")
    print(f"Extra observations keys: {extra_obs['observations'].keys()}")

    # Test action stepping
    actions = torch.randn(env.num_envs, env.num_actions)
    obs, rewards, dones, infos = env.step(actions)
    print(f"Step results - obs: {obs.shape}, rewards: {rewards.shape}, dones: {dones.shape}")

    # Test reset
    env.reset()
    print("Environment reset successfully")

    # Close environment
    env.close()
    print("Environment closed successfully")


if __name__ == "__main__":
    # Uncomment the function you want to run
    # main()  # For full training
    # test_wrapper()  # For testing the wrapper
    print("Example usage file created. Please adapt the scenario configuration to your needs.")
