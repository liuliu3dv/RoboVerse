import os
from isaaclab.sim.converters.urdf_converter import UrdfConverter, UrdfConverterCfg

def convert_urdf_to_usd(urdf_path: str, usd_dir: str, usd_name: str = None):
    cfg = UrdfConverterCfg(
        asset_path=os.path.abspath(urdf_path),
        usd_dir=os.path.abspath(usd_dir),
        usd_file_name=usd_name,
        fix_base=True,
        merge_fixed_joints=True,
        make_instanceable=True,
        force_usd_conversion=True,
        root_link_name=None,     # 可指定根 link 名称
        link_density=1000.0,
        convert_mimic_joints_to_normal_joints=False,
        joint_drive=None
    )
    conv = UrdfConverter(cfg)
    usd_path = conv.usd_path
    print(f"✅ URDF 转换完成：{urdf_path}  →  {usd_path}")
    return usd_path

if __name__ == "__main__":
    urdf = "r/home/RoboVerse/humanoid_retargeting/assets/table/table.urdf"
    out_dir = "/home/RoboVerse/humanoid_retargeting/assets/table/"
    convert_urdf_to_usd(urdf, out_dir, usd_name="table.usd")
