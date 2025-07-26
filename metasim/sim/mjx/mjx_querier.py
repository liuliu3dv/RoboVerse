# mjx_query_helper.py

import jax
import jax.numpy as jnp
import mujoco
import torch

from metasim.cfg.query_type import ContactForce, SitePos, SiteXMat, SensorData


FEET_SITES = [
    "left_foot",
    "right_foot",
]

HAND_SITES = [
    "left_palm",
    "right_palm",
]

LEFT_FEET_GEOMS = ["left_foot"]
RIGHT_FEET_GEOMS = ["right_foot"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

ROOT_BODY = "torso_link"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

class MJXQuerier:
    """
    Add new Query types by inserting into QUERY_MAP.
    """

    _site_cache = {}
    _body_cache = {}

    QUERY_MAP = {
        SitePos: "site_pos",
        ContactForce: "contact_force",
        # add here
    }

    # public entry ------------------------------------------------------------
    @classmethod
    def query(cls, q, handler, robot_name=None):
        fn_name = cls.QUERY_MAP[type(q)]
        return j2t(getattr(cls, fn_name)(q, handler, robot_name))

        # query func collection ---------------------------------------------------

    @classmethod
    def site_pos(cls, q: SitePos, handler, robot_name):
        """Return (N_env, 3) site position."""
        key = id(handler._mj_model)
        cache = cls._site_cache.setdefault(key, {})

        if not cache:
            cache.update({
                mujoco.mj_id2name(handler._mj_model, mujoco.mjtObj.mjOBJ_SITE, i): i
                for i in range(handler._mj_model.nsite)
                if mujoco.mj_id2name(handler._mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
            })

        full_name = f"{robot_name}/{q.site}" if "/" not in q.site else q.site
        sid = cache[full_name]

        return handler._data.site_xpos[:, sid]

    @classmethod
    def contact_force(cls, q: ContactForce, handler, robot_name):
        """Return (N_env, 6) force torque of one body."""
        key = id(handler._mj_model)
        cache = cls._body_cache.setdefault(key, {})
        if not cache:
            cache.update({
                mujoco.mj_id2name(handler._mj_model, mujoco.mjtObj.mjOBJ_BODY, i): i
                for i in range(handler._mj_model.nbody)
                if mujoco.mj_id2name(handler._mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            })

        full_name = f"{robot_name}/{q.sensor_name}" if "/" not in q.site else q.site
        bid = cache[full_name]

        contact_force = handler._data.cfrc_ext[:, jnp.asarray([bid], jnp.int32)]
        return contact_force[:, 0, :]  # (N_env, 6)


    @classmethod
    def site_xmat(cls, q: SiteXMat, handler):
        """Return (N_env, 9) flattened site rotation matrix."""
        mdl, dat = handler._mj_model, handler._data
        key = id(mdl)
        cache = cls._site_cache.setdefault(key, {})

        if q.name not in cache:
            cache[q.name] = mdl.site(q.name).id           # cache site id
        sid = cache[q.name]

        # dat.site_xmat shape: (N_env, N_site, 9)
        return dat.site_xmat[:, sid]   

    @classmethod
    def sensor(cls, q: SensorData, handler):
        """Return (N_env, dim) sensor value."""
        mdl, dat = handler._mj_model, handler._data
        key = id(mdl)
        cache = cls._sensor_cache.setdefault(key, {})

        if q.name not in cache:
            sid  = mdl.sensor(q.name).id
            cache[q.name] = (mdl.sensor_adr[sid], mdl.sensor_dim[sid])

        adr, dim = cache[q.name]
        return dat.sensordata[:, jnp.arange(adr, adr + dim)]  # (N_env, dim)
    
def j2t(a: jax.Array, device="cuda") -> torch.Tensor:
    """Convert a JAX array to a PyTorch tensor, keeping it on the requested device."""
    if device:
        tgt = torch.device(device)
        plat = "gpu" if tgt.type == "cuda" else tgt.type
        if a.device.platform != plat:
            a = jax.device_put(a, jax.devices(plat)[tgt.index or 0])
    return torch.from_dlpack(jax.dlpack.to_dlpack(a))


# -----------------------------------------------------------------------------
# usage in handler
# -----------------------------------------------------------------------------
# value = MJXQuerier.query(query_obj, handler)
