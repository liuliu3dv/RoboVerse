"""This module contrains GraspNet tasks."""

from .graspnet_task_cfg import _dynamic_import_task


def __getattr__(name):
    if name.startswith("GraspNet") and name.endswith("Cfg"):
        # task_name = name[:-7]  # why -7? 0000 is removed
        task_name = name[:-3]
        return _dynamic_import_task(task_name)
    else:
        raise AttributeError(f"Module {__name__} has no attribute {name}")
