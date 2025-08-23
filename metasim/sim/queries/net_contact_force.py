from __future__ import annotations
from metasim.queries.base import BaseQueryType


class NetContactForce(BaseQueryType):
    def __init__(self):
        super().__init__()

    def bind_handler(self, handler, *args, **kwargs):
        """Remember the site-id once the handler is known."""
        super().bind_handler(handler, *args, **kwargs)
        mod = handler.__class__.__module__

        if mod.startswith("metasim.sim.isaacsim"):
            pass
        else:
            raise ValueError(f"Unsupported handler type: {type(handler)} for SitePos query")

    def __call__(self):
        """Return (num_env, num_bodies) contact force whenever `get_extra()` is invoked."""
        mod = self.handler.__class__.__module__

        if mod.startswith("metasim.sim.isaacsim"):
            robot_name = self.handler.robots[0].name
            reindex = self.handler.get_body_reindex(robot_name)
            return self.handler.contact_sensor.data.net_forces_w[:, reindex, :]
        else:
            raise ValueError(f"Unsupported handler type: {type(self.handler)} for NetContactForce query")
