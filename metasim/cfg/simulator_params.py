"""Configuration classes for simulator parameters."""

from __future__ import annotations

from metasim.utils import configclass


@configclass
class SimParamCfg:
    """Simulation parameters cfg.

    This class defines the parameters for the simulator.
    It is important to ensure that each task is configured with appropriate simulation
    parameters to avoid divergence or unexpected results.

    Reference for IsaacGym: https://docs.robotsfan.com/isaacgym/api/python/struct_py.html#isaacgym.gymapi.PhysXParams
    """

    ## Simulation
    dt: float | None = None
    """The time-step of the simulation. If None, the default value in the original simulator will be used. The default value for each simulator is:
    - IsaacGym: 1/60
    - IsaacLab: 1/60
    - MuJoCo: 1/500
    - PyBullet: 1/240
    """

    ## Physics
    bounce_threshold_velocity: float = 0.2
    contact_offset: float = 0.001
    num_position_iterations: int = 8
    num_velocity_iterations: int = 1
    friction_correlation_distance: float = 0.0005
    friction_offset_threshold: float = 0.001
    replace_cylinder_with_capsule: bool = False
    rest_offset: float = 0.0
    solver_type: int = 1
    substeps: int = 1  # for IsaacGym
    max_depenetration_velocity: float = 1.0
    default_buffer_size_multiplier: int = 2.0

    ## Resource management
    num_threads: int = 0
    # XXX: these parameters should be replaced by "device" in the future
    use_gpu_pipeline: bool = True
    use_gpu: bool = True
