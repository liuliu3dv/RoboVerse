from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from shutil import copy

import trimesh
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = [
    "AssetConverterFactory",
    "AssetType",
    "MeshtoMJCFConverter",
    "MeshtoUSDConverter",
    "URDFtoUSDConverter",
]


@dataclass
class AssetType(str):
    """Asset type enumeration."""

    MJCF = "mjcf"
    USD = "usd"
    URDF = "urdf"
    MESH = "mesh"


class AssetConverterBase(ABC):
    """Converter abstract base class."""

    @abstractmethod
    def convert(self, urdf_path: str, output_path: str, **kwargs) -> str:
        pass

    def transform_mesh(self, input_mesh: str, output_mesh: str, mesh_origin: ET.Element) -> None:
        """Apply transform to the mesh based on the origin element in URDF."""
        mesh = trimesh.load(input_mesh)
        rpy = list(map(float, mesh_origin.get("rpy").split(" ")))
        rotation = Rotation.from_euler("xyz", rpy, degrees=False)
        offset = list(map(float, mesh_origin.get("xyz").split(" ")))
        mesh.vertices = (mesh.vertices @ rotation.as_matrix().T) + offset

        os.makedirs(os.path.dirname(output_mesh), exist_ok=True)
        _ = mesh.export(output_mesh)

        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class MeshtoMJCFConverter(AssetConverterBase):
    """Convert URDF files into MJCF format."""

    def __init__(
        self,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs

    def _copy_asset_file(self, src: str, dst: str) -> None:
        if os.path.exists(dst):
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        copy(src, dst)

    def add_geometry(
        self,
        mujoco_element: ET.Element,
        link: ET.Element,
        body: ET.Element,
        tag: str,
        input_dir: str,
        output_dir: str,
        mesh_name: str,
        material: ET.Element | None = None,
        is_collision: bool = False,
    ) -> None:
        """Add geometry to the MJCF body from the URDF link."""
        element = link.find(tag)
        geometry = element.find("geometry")
        mesh = geometry.find("mesh")
        filename = mesh.get("filename")
        scale = mesh.get("scale", "1.0 1.0 1.0")

        mesh_asset = ET.SubElement(mujoco_element, "mesh", name=mesh_name, file=filename, scale=scale)
        geom = ET.SubElement(body, "geom", type="mesh", mesh=mesh_name)

        self._copy_asset_file(
            f"{input_dir}/{filename}",
            f"{output_dir}/{filename}",
        )

        # Preprocess the mesh by applying rotation.
        input_mesh = f"{input_dir}/{filename}"
        output_mesh = f"{output_dir}/{filename}"
        mesh_origin = element.find("origin")
        if mesh_origin is not None:
            self.transform_mesh(input_mesh, output_mesh, mesh_origin)

        if material is not None:
            geom.set("material", material.get("name"))

        if is_collision:
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            geom.set("rgba", "1 1 1 0")

    def add_materials(
        self,
        mujoco_element: ET.Element,
        link: ET.Element,
        tag: str,
        input_dir: str,
        output_dir: str,
        name: str,
        reflectance: float = 0.2,
    ) -> ET.Element:
        """Add materials to the MJCF asset from the URDF link."""
        element = link.find(tag)
        geometry = element.find("geometry")
        mesh = geometry.find("mesh")
        filename = mesh.get("filename")
        dirname = os.path.dirname(filename)

        material = ET.SubElement(
            mujoco_element,
            "material",
            name=f"material_{name}",
            texture=f"texture_{name}",
            reflectance=str(reflectance),
        )
        ET.SubElement(
            mujoco_element,
            "texture",
            name=f"texture_{name}",
            type="2d",
            file=f"{dirname}/material_0.png",
        )

        self._copy_asset_file(
            f"{input_dir}/{dirname}/material_0.png",
            f"{output_dir}/{dirname}/material_0.png",
        )

        return material

    def convert(self, urdf_path: str, mjcf_path: str):
        """Convert a URDF file to MJCF format."""
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        mujoco_struct = ET.Element("mujoco")
        mujoco_struct.set("model", root.get("name"))
        mujoco_asset = ET.SubElement(mujoco_struct, "asset")
        mujoco_worldbody = ET.SubElement(mujoco_struct, "worldbody")

        input_dir = os.path.dirname(urdf_path)
        output_dir = os.path.dirname(mjcf_path)
        os.makedirs(output_dir, exist_ok=True)
        for idx, link in enumerate(root.findall("link")):
            link_name = link.get("name", "unnamed_link")
            body = ET.SubElement(mujoco_worldbody, "body", name=link_name)

            material = self.add_materials(
                mujoco_asset,
                link,
                "visual",
                input_dir,
                output_dir,
                name=str(idx),
            )
            self.add_geometry(
                mujoco_asset,
                link,
                body,
                "visual",
                input_dir,
                output_dir,
                f"visual_mesh_{idx}",
                material,
            )
            self.add_geometry(
                mujoco_asset,
                link,
                body,
                "collision",
                input_dir,
                output_dir,
                f"collision_mesh_{idx}",
                is_collision=True,
            )

        tree = ET.ElementTree(mujoco_struct)
        ET.indent(tree, space="  ", level=0)

        tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)
        logger.info(f"Successfully converted {urdf_path} → {mjcf_path}")


class MeshtoUSDConverter(AssetConverterBase):
    """Convert Mesh file from URDF into USD format."""

    DEFAULT_BIND_APIS = [
        "MaterialBindingAPI",
        "PhysicsMeshCollisionAPI",
        "PhysicsCollisionAPI",
        "PhysxCollisionAPI",
        "PhysicsMassAPI",
        "PhysicsRigidBodyAPI",
        "PhysxRigidBodyAPI",
    ]

    def __init__(
        self,
        force_usd_conversion: bool = True,
        make_instanceable: bool = False,
        simulation_app=None,
        **kwargs,
    ):
        self.usd_parms = dict(
            force_usd_conversion=force_usd_conversion,
            make_instanceable=make_instanceable,
            **kwargs,
        )
        if simulation_app is not None:
            self.simulation_app = simulation_app

    def __enter__(self):
        from isaaclab.app import AppLauncher

        if not hasattr(self, "simulation_app"):
            launch_args = dict(
                headless=True,
                no_splash=True,
                fast_shutdown=True,
                disable_gpu=True,
            )
            self.app_launcher = AppLauncher(launch_args)
            self.simulation_app = self.app_launcher.app

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the simulation app if it was created here
        if hasattr(self, "app_launcher"):
            self.simulation_app.close()

        if exc_val is not None:
            logger.error(f"Exception occurred: {exc_val}.")

        return False

    def convert(self, urdf_path: str, output_file: str):
        """Convert a URDF file to USD and post-process collision meshes."""
        from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
        from pxr import PhysxSchema, Sdf, Usd, UsdShade

        tree = ET.parse(urdf_path)
        root = tree.getroot()
        mesh_file = root.find("link/visual/geometry/mesh").get("filename")
        input_mesh = os.path.join(os.path.dirname(urdf_path), mesh_file)
        output_dir = os.path.abspath(os.path.dirname(output_file))
        output_mesh = f"{output_dir}/mesh/{os.path.basename(mesh_file)}"
        mesh_origin = root.find("link/visual/origin")
        if mesh_origin is not None:
            self.transform_mesh(input_mesh, output_mesh, mesh_origin)

        cfg = MeshConverterCfg(
            asset_path=output_mesh,
            usd_dir=output_dir,
            usd_file_name=os.path.basename(output_file),
            **self.usd_parms,
        )
        urdf_converter = MeshConverter(cfg)
        usd_path = urdf_converter.usd_path

        stage = Usd.Stage.Open(usd_path)
        layer = stage.GetRootLayer()
        with Usd.EditContext(stage, layer):
            for prim in stage.Traverse():
                # Change texture path to relative path.
                if prim.GetName() == "material_0":
                    shader = UsdShade.Shader(prim).GetInput("diffuse_texture")
                    if shader.Get() is not None:
                        relative_path = shader.Get().path.replace(f"{output_dir}/", "")
                        shader.Set(Sdf.AssetPath(relative_path))

                # Add convex decomposition collision and set ShrinkWrap.
                elif prim.GetName() == "mesh":
                    approx_attr = prim.GetAttribute("physics:approximation")
                    if not approx_attr:
                        approx_attr = prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token)
                    approx_attr.Set("convexDecomposition")

                    physx_conv_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                    physx_conv_api.GetShrinkWrapAttr().Set(True)

                    api_schemas = prim.GetMetadata("apiSchemas")
                    if api_schemas is None:
                        api_schemas = Sdf.TokenListOp()

                    api_list = list(api_schemas.GetAddedOrExplicitItems())
                    for api in self.DEFAULT_BIND_APIS:
                        if api not in api_list:
                            api_list.append(api)

                    api_schemas.appendedItems = api_list
                    prim.SetMetadata("apiSchemas", api_schemas)

        layer.Save()
        logger.info(f"Successfully converted {urdf_path} → {usd_path}")


class URDFtoUSDConverter(MeshtoUSDConverter):
    """Convert URDF files into USD format.

    Args:
        fix_base (bool): Whether to fix the base link.
        merge_fixed_joints (bool): Whether to merge fixed joints.
        make_instanceable (bool): Whether to make prims instanceable.
        force_usd_conversion (bool): Force conversion to USD.
        collision_from_visuals (bool): Generate collisions from visuals if not provided.
    """

    def __init__(
        self,
        fix_base: bool = False,
        merge_fixed_joints: bool = False,
        make_instanceable: bool = True,
        force_usd_conversion: bool = True,
        collision_from_visuals: bool = True,
        joint_drive=None,
        rotate_wxyz: tuple[float] | None = None,
        simulation_app=None,
        **kwargs,
    ):
        self.usd_parms = dict(
            fix_base=fix_base,
            merge_fixed_joints=merge_fixed_joints,
            make_instanceable=make_instanceable,
            force_usd_conversion=force_usd_conversion,
            collision_from_visuals=collision_from_visuals,
            joint_drive=joint_drive,
            **kwargs,
        )
        self.rotate_wxyz = rotate_wxyz
        if simulation_app is not None:
            self.simulation_app = simulation_app

    def convert(self, urdf_path: str, output_file: str):
        """Convert a URDF file to USD and post-process collision meshes."""
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom

        cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=os.path.abspath(os.path.dirname(output_file)),
            usd_file_name=os.path.basename(output_file),
            **self.usd_parms,
        )

        urdf_converter = UrdfConverter(cfg)
        usd_path = urdf_converter.usd_path

        stage = Usd.Stage.Open(usd_path)
        layer = stage.GetRootLayer()
        with Usd.EditContext(stage, layer):
            for prim in stage.Traverse():
                if prim.GetName() == "collisions":
                    approx_attr = prim.GetAttribute("physics:approximation")
                    if not approx_attr:
                        approx_attr = prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token)
                    approx_attr.Set("convexDecomposition")

                    physx_conv_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                    physx_conv_api.GetShrinkWrapAttr().Set(True)

                    api_schemas = prim.GetMetadata("apiSchemas")
                    if api_schemas is None:
                        api_schemas = Sdf.TokenListOp()

                    api_list = list(api_schemas.GetAddedOrExplicitItems())
                    for api in self.DEFAULT_BIND_APIS:
                        if api not in api_list:
                            api_list.append(api)

                    api_schemas.appendedItems = api_list
                    prim.SetMetadata("apiSchemas", api_schemas)

        if self.rotate_wxyz is not None:
            inner_prim = next(p for p in stage.GetDefaultPrim().GetChildren() if p.IsA(UsdGeom.Xform))
            xformable = UsdGeom.Xformable(inner_prim)
            xformable.ClearXformOpOrder()
            orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            orient_op.Set(Gf.Quatd(*self.rotate_wxyz))

        layer.Save()
        logger.info(f"Successfully converted {urdf_path} → {usd_path}")


class AssetConverterFactory:
    """Factory class for creating asset converters based on target and source types."""

    @staticmethod
    def create(target_type: AssetType, source_type: AssetType = "urdf", **kwargs) -> AssetConverterBase:
        """Create an asset converter instance based on target and source types."""
        if target_type == AssetType.MJCF and source_type == AssetType.URDF:
            converter = MeshtoMJCFConverter(**kwargs)
        elif target_type == AssetType.USD and source_type == AssetType.URDF:
            converter = URDFtoUSDConverter(**kwargs)
        elif target_type == AssetType.USD and source_type == AssetType.MESH:
            converter = MeshtoUSDConverter(**kwargs)
        else:
            raise ValueError(f"Unsupported converter type: {source_type} -> {target_type}.")

        return converter


if __name__ == "__main__":
    # target_asset_type = AssetType.MJCF
    target_asset_type = AssetType.USD

    urdf_paths = [
        "outputs/layouts_gens/task_0002/asset3d/apple/result/apple.urdf",
        "outputs/layouts_gens/task_0002/asset3d/banana/result/banana.urdf",
        "outputs/layouts_gens/task_0002/asset3d/mug/result/mug.urdf",
        "outputs/layouts_gens/task_0002/asset3d/napkin/result/napkin.urdf",
        "outputs/layouts_gens/task_0002/asset3d/plate/result/plate.urdf",
        "outputs/layouts_gens/task_0002/asset3d/table/result/table.urdf",
    ]

    if target_asset_type == AssetType.MJCF:
        output_files = [
            "outputs/layouts_gens/task_0000/mujoco2/apple/apple.mjcf",
            "outputs/layouts_gens/task_0000/mujoco2/banana/banana.mjcf",
            "outputs/layouts_gens/task_0000/mujoco2/mug/mug.mjcf",
            "outputs/layouts_gens/task_0000/mujoco2/napkin/napkin.mjcf",
            "outputs/layouts_gens/task_0000/mujoco2/plate/plate.mjcf",
            "outputs/layouts_gens/task_0000/mujoco2/table/table.mjcf",
        ]
        asset_converter = AssetConverterFactory.create(
            target_type=AssetType.MJCF,
            source_type=AssetType.URDF,
        )

    elif target_asset_type == AssetType.USD:
        output_files = [
            "outputs/layouts_gens/task_0000/isaac2/apple/apple.usd",
            "outputs/layouts_gens/task_0000/isaac2/banana/banana.usd",
            "outputs/layouts_gens/task_0000/isaac2/mug/mug.usd",
            "outputs/layouts_gens/task_0000/isaac2/napkin/napkin.usd",
            "outputs/layouts_gens/task_0000/isaac2/plate/plate.usd",
            "outputs/layouts_gens/task_0000/isaac2/table/table.usd",
        ]
        asset_converter = AssetConverterFactory.create(
            target_type=AssetType.USD,
            source_type=AssetType.MESH,
        )

    with asset_converter:
        for urdf_path, output_file in zip(urdf_paths, output_files):
            asset_converter.convert(urdf_path, output_file)

    # urdf_path = "outputs/layouts_gens/task_0000/asset3d/desk/result/desk.urdf"
    # output_file = "outputs/layouts_gens/task_0000/asset3d/desk/test_cvt_usd/desk.usd"

    # asset_converter = AssetConverterFactory.create(
    #     target_type=AssetType.USD,
    #     source_type=AssetType.URDF,
    #     rotate_wxyz=(0.7071, 0.7071, 0, 0),  # rotate 90 deg around the X-axis
    # )

    # with asset_converter:
    #     asset_converter.convert(urdf_path, output_file)
