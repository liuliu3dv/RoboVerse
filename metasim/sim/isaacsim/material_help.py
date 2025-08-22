"""Material handling helper for Isaac Sim.

This module contains all material-related functionality extracted from the main
IsaacSim class to improve code organization and maintainability.
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
from loguru import logger as log


class MaterialHelper:
    """Helper class for applying various types of materials to objects in Isaac Sim."""

    def __init__(self, sim_instance: Any):
        """Initialize the material helper.

        Args:
            sim_instance: Reference to the main IsaacSim instance for accessing
                         device, num_envs, scene, etc.
        """
        self.sim = sim_instance

    def apply_materials(self, material_assignments: dict) -> None:
        """Apply material assignments to objects."""
        log.debug("Applying materials from randomizer")

        # Keep track of objects that have materials applied
        objects_with_materials = set()
        objects_with_physics_materials = set()

        # Apply object materials
        object_materials = material_assignments.get("object_materials", [])
        for assignment in object_materials:
            obj_name = getattr(assignment["object"], "name", None)
            if obj_name:
                objects_with_materials.add(obj_name)
            self._apply_material_to_object(assignment["object"], assignment["material_config"], assignment["category"])

        # Apply environment materials
        environment_materials = material_assignments.get("environment_materials", [])
        for assignment in environment_materials:
            obj_name = getattr(assignment["object"], "name", None)
            if obj_name:
                objects_with_materials.add(obj_name)
            self._apply_material_to_object(assignment["object"], assignment["material_config"], assignment["category"])

        # Apply robot materials
        robot_materials = material_assignments.get("robot_materials", [])
        for assignment in robot_materials:
            obj_name = getattr(assignment["object"], "name", None)
            if obj_name:
                objects_with_materials.add(obj_name)
            self._apply_material_to_object(assignment["object"], assignment["material_config"], assignment["category"])

        # Apply physics materials
        physics_materials = material_assignments.get("physics_materials", [])
        for assignment in physics_materials:
            obj_name = getattr(assignment["object"], "name", None)
            if obj_name:
                objects_with_physics_materials.add(obj_name)
            self._apply_physics_material_to_object(assignment["object"], assignment["material_config"])

        # Reset materials for objects that weren't assigned any materials (neither PBR nor physics)
        all_objects_with_any_materials = objects_with_materials | objects_with_physics_materials
        self._reset_unassigned_object_materials(all_objects_with_any_materials)

        log.debug(
            f"Applied {len(object_materials)} object materials, "
            f"{len(environment_materials)} environment materials, "
            f"{len(robot_materials)} robot materials, "
            f"{len(physics_materials)} physics materials"
        )

    def _apply_material_to_object(self, obj: Any, material_config: dict, category: str) -> bool:
        """Apply a material configuration to an object."""
        try:
            material_type = material_config.get("type", "pbr")

            if material_type == "mdl":
                return self._apply_mdl_material(obj, material_config)
            elif material_type in ["pbr", "preview_surface"]:
                return self._apply_pbr_material(obj, material_config)
            else:
                log.warning(f"Unknown material type: {material_type}")
                return False

        except Exception as e:
            log.warning(f"Failed to apply material to {category} object {getattr(obj, 'name', 'unnamed')}: {e}")
            # Fallback: simple color randomization
            return self._apply_simple_color_randomization(obj)

    def _apply_mdl_material(self, obj: Any, material_config: dict) -> bool:
        """Apply an MDL material to an object with robust error handling."""
        try:
            # Get object's USD prim path
            prim_path = self._get_object_prim_path(obj)
            if not prim_path:
                log.debug(f"Could not get prim path for object {getattr(obj, 'name', 'unnamed')}, falling back to PBR")
                return self._apply_pbr_material_fallback(obj, material_config)

            mdl_path = material_config.get("mdl_path")
            if not mdl_path:
                log.debug(f"No MDL path specified for {getattr(obj, 'name', 'unnamed')}, falling back to PBR")
                return self._apply_pbr_material_fallback(obj, material_config)

            # Validate MDL file exists
            if not os.path.exists(mdl_path):
                log.debug(f"MDL file not found: {mdl_path}, attempting to download...")
                try:
                    # Try to download the file using the same system as IsaacLab
                    from metasim.utils.hf_util import check_and_download_single

                    check_and_download_single(mdl_path)
                    log.debug(f"Downloaded MDL file: {mdl_path}")
                except Exception as download_error:
                    log.debug(f"Failed to download MDL file {mdl_path}: {download_error}")
                    # Fallback to PBR
                    return self._apply_pbr_material_fallback(obj, material_config)

            # Final validation that file exists and is readable
            if not self._validate_mdl_file(mdl_path):
                log.debug(f"MDL file validation failed for {mdl_path}, falling back to PBR")
                return self._apply_pbr_material_fallback(obj, material_config)

            # Also download associated texture files referenced in the MDL
            try:
                # Get the directory containing the MDL file
                mdl_dir = os.path.dirname(mdl_path)
                texture_dir = os.path.join(mdl_dir, "textures")

                # Read MDL file to find texture references
                if os.path.exists(mdl_path):
                    with open(mdl_path, encoding="utf-8", errors="ignore") as f:
                        mdl_content = f.read()

                    # Extract texture file references from MDL content
                    # Look for patterns like: texture_2d("./textures/filename.jpg")
                    texture_patterns = [
                        r'texture_2d\s*\(\s*["\']([^"\']+)["\']',
                        r'file\s*:\s*["\']([^"\']+\.(?:jpg|png|tif|tiff|exr|hdr))["\']',
                        r'["\']([^"\']*textures[^"\']*\.(?:jpg|png|tif|tiff|exr|hdr))["\']',
                    ]

                    from metasim.utils.hf_util import check_and_download_single

                    texture_files = set()

                    for pattern in texture_patterns:
                        matches = re.findall(pattern, mdl_content, re.IGNORECASE)
                        for match in matches:
                            # Convert relative paths to absolute
                            if match.startswith("./"):
                                texture_path = os.path.join(mdl_dir, match[2:])
                            elif not os.path.isabs(match):
                                texture_path = os.path.join(mdl_dir, match)
                            else:
                                texture_path = match
                            texture_files.add(texture_path)

                    # Download each texture file
                    for texture_path in texture_files:
                        if not os.path.exists(texture_path):
                            try:
                                check_and_download_single(texture_path)
                                log.debug(f"Downloaded texture: {texture_path}")
                            except Exception as tex_error:
                                log.debug(f"Could not download texture {texture_path}: {tex_error}")

            except Exception as e:
                log.debug(f"Could not download textures for {mdl_path}: {e}")

            # Find material exports
            exports = re.findall(r"export\s+material\s+([A-Za-z_]\w*)\s*\(", mdl_content)
            mdl_export = material_config.get("mdl_export")
            if not mdl_export:
                mdl_export = exports[0] if exports else None

            if not mdl_export:
                log.debug(f"No export material found in {mdl_path}, falling back to PBR")
                return self._apply_pbr_material_fallback(obj, material_config)

            material_name = material_config.get("name", mdl_export)
            unique_material_name = f"{material_name}_{id(material_config)}"
            material_path = f"/World/Looks/{unique_material_name}"

            # Create MDL material in USD stage
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if not stage:
                log.warning("No USD stage available for MDL material creation")
                return False

            # Check if material already exists
            material_prim = stage.GetPrimAtPath(material_path)
            if material_prim.IsValid():
                log.debug(f"MDL material already exists at {material_path}, reusing")
            else:
                # Create MDL material using Isaac Lab commands with enhanced error handling
                try:
                    success, result = omni.kit.commands.execute(
                        "CreateMdlMaterialPrim",
                        mtl_url=mdl_path,
                        mtl_name=mdl_export,
                        mtl_path=material_path,
                        select_new_prim=False,
                    )
                    if not success:
                        log.debug(f"Failed to create MDL material at {material_path}, falling back to PBR")
                        return self._apply_pbr_material_fallback(obj, material_config)

                    # Verify the material was created successfully
                    material_prim = stage.GetPrimAtPath(material_path)
                    if not material_prim.IsValid():
                        log.debug(f"MDL material prim invalid after creation at {material_path}, falling back to PBR")
                        return self._apply_pbr_material_fallback(obj, material_config)

                    log.debug(f"Successfully created MDL material at {material_path}")

                except Exception as e:
                    log.debug(f"Failed to create MDL material at {material_path}: {e}, falling back to PBR")
                    return self._apply_pbr_material_fallback(obj, material_config)

            # Bind MDL material to all environment instances
            success_count = 0
            for env_idx in range(self.sim.num_envs):
                env_prim_path = prim_path.replace("env_0", f"env_{env_idx}")
                if self._bind_material_to_prim(env_prim_path, material_path, "mdl"):
                    success_count += 1

            if success_count > 0:
                log.debug(
                    f"Applied MDL material '{unique_material_name}' to {getattr(obj, 'name', 'unnamed')} (bindings: {success_count}/{self.sim.num_envs})"
                )
                return True
            else:
                log.debug(f"MDL material binding failed for {getattr(obj, 'name', 'unnamed')}, falling back to PBR")
                return self._apply_pbr_material_fallback(obj, material_config)

        except Exception as e:
            log.debug(f"Failed to apply MDL material: {e}, falling back to PBR")
            return self._apply_pbr_material_fallback(obj, material_config)

    def _validate_mdl_file(self, mdl_path: str) -> bool:
        """Validate that an MDL file exists and is readable."""
        try:
            if not os.path.exists(mdl_path):
                return False

            # Check if file is readable
            if not os.access(mdl_path, os.R_OK):
                return False

            # Basic file size check (empty files are probably invalid)
            if os.path.getsize(mdl_path) < 100:  # Very small threshold
                return False

            # Try to read a few lines to ensure it's not corrupted
            try:
                with open(mdl_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read(1000)  # Read first 1KB
                    if "mdl" not in content.lower():
                        return False
            except Exception:
                # If we can't read it as text, it might still be a valid binary MDL
                pass

            return True
        except Exception as e:
            log.debug(f"MDL file validation failed for {mdl_path}: {e}")
            return False

    def _apply_pbr_material_fallback(self, obj: Any, material_config: dict) -> bool:
        """Apply a PBR material as fallback when MDL fails."""
        try:
            # Create a PBR material config based on the MDL config
            pbr_config = {
                "type": "pbr",
                "name": material_config.get("name", "fallback_pbr"),
                "diffuse_color": material_config.get("diffuse_color", [0.7, 0.7, 0.7]),
                "metallic": material_config.get("metallic", 0.0),
                "roughness": material_config.get("roughness", 0.5),
                "opacity": material_config.get("opacity", 1.0),
            }

            # Try to infer material properties from MDL name
            material_name = material_config.get("name", "").lower()
            if "metal" in material_name or "aluminum" in material_name:
                pbr_config["metallic"] = 0.8
                pbr_config["roughness"] = 0.1
                pbr_config["diffuse_color"] = [0.8, 0.8, 0.9]
            elif "wood" in material_name:
                pbr_config["metallic"] = 0.0
                pbr_config["roughness"] = 0.7
                pbr_config["diffuse_color"] = [0.6, 0.4, 0.2]
            elif "glass" in material_name:
                pbr_config["metallic"] = 0.0
                pbr_config["roughness"] = 0.0
                pbr_config["opacity"] = 0.7
                pbr_config["diffuse_color"] = [0.9, 0.9, 1.0]

            log.debug(f"Applying PBR fallback for {getattr(obj, 'name', 'unnamed')} with inferred properties")
            return self._apply_pbr_material(obj, pbr_config)

        except Exception as e:
            log.debug(f"PBR fallback failed for {getattr(obj, 'name', 'unnamed')}: {e}")
            return self._apply_simple_color_randomization(obj)

    def _apply_pbr_material(self, obj: Any, material_config: dict) -> bool:
        """Apply a PBR material to an object by creating and binding USD material."""
        try:
            # Get object's USD prim path
            prim_path = self._get_object_prim_path(obj)
            if not prim_path:
                log.warning(f"Could not determine prim path for object {getattr(obj, 'name', 'unnamed')}")
                return False

            # Create unique material name and path
            material_name = material_config.get("name", "pbr_material")
            unique_material_name = f"{material_name}_{id(obj)}"
            material_path = f"/World/Looks/{unique_material_name}"

            # Import Isaac Lab material utilities
            from isaaclab.sim.spawners.materials import visual_materials_cfg

            # Extract PBR properties with variation
            diffuse_color = material_config.get("diffuse_color", [0.5, 0.5, 0.5])
            if len(diffuse_color) == 3:
                # Add small color variation
                color_var = 0.1
                diffuse_color = [
                    float(np.clip(c + np.random.uniform(-color_var, color_var), 0.0, 1.0)) for c in diffuse_color
                ]

            emissive_color = material_config.get("emissive_color", [0.0, 0.0, 0.0])
            if len(emissive_color) == 3:
                emissive_color = tuple(emissive_color)
            else:
                emissive_color = (0.0, 0.0, 0.0)

            # Extract other PBR properties with small variations
            roughness = float(material_config.get("roughness", 0.5))
            roughness = np.clip(roughness + np.random.uniform(-0.05, 0.05), 0.0, 1.0)

            metallic = float(material_config.get("metallic", 0.0))
            metallic = np.clip(metallic + np.random.uniform(-0.05, 0.05), 0.0, 1.0)

            opacity = float(material_config.get("opacity", 1.0))
            opacity = np.clip(opacity + np.random.uniform(-0.05, 0.05), 0.0, 1.0)

            # Create PreviewSurface material configuration
            material_cfg = visual_materials_cfg.PreviewSurfaceCfg(
                diffuse_color=tuple(diffuse_color),
                emissive_color=emissive_color,
                roughness=roughness,
                metallic=metallic,
                opacity=opacity,
            )

            # Create the material in USD stage
            try:
                # Ensure the /World/Looks directory exists
                import omni.usd

                stage = omni.usd.get_context().get_stage()
                if stage:
                    looks_prim = stage.GetPrimAtPath("/World/Looks")
                    if not looks_prim.IsValid():
                        from pxr import UsdGeom

                        looks_prim = UsdGeom.Scope.Define(stage, "/World/Looks")
                        log.debug("Created /World/Looks scope")

                # Check if material already exists
                material_prim = stage.GetPrimAtPath(material_path)
                if material_prim.IsValid():
                    log.debug(f"PBR material already exists at {material_path}, reusing")
                else:
                    material_cfg.func(material_path, material_cfg)
                    log.debug(f"Successfully created PBR material at {material_path}")

                    # Verify the material was created
                    material_prim = stage.GetPrimAtPath(material_path)
                    if not material_prim.IsValid():
                        log.warning(f"Material prim not found at {material_path} after creation")
                        return False
                    else:
                        log.debug(f"Material prim verified at {material_path}")

            except Exception as e:
                log.warning(f"Failed to create PBR material at {material_path}: {e}")
                return False

            # Bind PBR material to all environment instances
            success_count = 0
            for env_idx in range(self.sim.num_envs):
                env_prim_path = prim_path.replace("env_0", f"env_{env_idx}")
                if self._bind_material_to_prim(env_prim_path, material_path, "visual"):
                    success_count += 1

            if success_count > 0:
                log.debug(
                    f"Applied PBR material '{unique_material_name}' to {getattr(obj, 'name', 'unnamed')} (bindings: {success_count}/{self.sim.num_envs})"
                )
                return True
            else:
                log.debug(f"PBR material binding failed for {getattr(obj, 'name', 'unnamed')}")
                return False

        except Exception as e:
            log.warning(f"Failed to apply PBR material to {getattr(obj, 'name', 'unnamed')}: {e}")
            # Fallback to simple color randomization
            return self._apply_simple_color_randomization(obj)

    def _apply_physics_material_to_object(self, obj: Any, physics_config: dict) -> bool:
        """Apply a physics material to an object."""
        try:
            # Get object's USD prim path - use first environment instance
            prim_path = self._get_object_prim_path(obj)
            if not prim_path:
                log.debug(f"Cannot get prim path for object {getattr(obj, 'name', 'unnamed')}")
                return False

            # Resolve path pattern to actual path
            if "env_.*" in prim_path:
                prim_path = prim_path.replace("env_.*", "env_0")

            # Create physics material using Isaac Lab
            import omni.usd
            from isaaclab.sim.spawners import materials as isaac_materials

            physics_cfg = isaac_materials.RigidBodyMaterialCfg(
                static_friction=float(physics_config.get("static_friction", 0.5)),
                dynamic_friction=float(physics_config.get("dynamic_friction", 0.5)),
                restitution=float(physics_config.get("restitution", 0.0)),
                friction_combine_mode=physics_config.get("friction_combine_mode", "average"),
                restitution_combine_mode=physics_config.get("restitution_combine_mode", "average"),
            )

            # Create unique material path
            material_name = physics_config.get("name", "physics_material")
            unique_material_name = f"{material_name}_{id(physics_config)}"
            material_path = f"/World/PhysicsMaterials/{unique_material_name}"

            # Create the physics material (check if it already exists first)
            stage = omni.usd.get_context().get_stage()
            if stage:
                material_prim = stage.GetPrimAtPath(material_path)
                if material_prim.IsValid():
                    log.debug(f"Physics material already exists at {material_path}, reusing")
                else:
                    physics_cfg.func(material_path, physics_cfg)
                    log.debug(f"Successfully created physics material at {material_path}")
            else:
                physics_cfg.func(material_path, physics_cfg)

            # Check if the object has physics properties before binding
            stage = omni.usd.get_context().get_stage()
            if stage:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    log.debug(f"Prim {prim_path} does not exist, cannot apply physics material")
                    return False

                # Check if object is part of an articulation (these need special handling)
                from pxr import PhysxSchema, UsdPhysics

                has_articulation = prim.HasAPI(UsdPhysics.ArticulationRootAPI)

                # Also check if it's a descendant of an articulation
                if not has_articulation:
                    parent_prim = prim.GetParent()
                    while parent_prim and parent_prim.GetPath() != "/":
                        if parent_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                            has_articulation = True
                            break
                        parent_prim = parent_prim.GetParent()

                # Skip physics materials for articulation objects (they need special configuration)
                if has_articulation:
                    log.debug(
                        f"Object {getattr(obj, 'name', 'unnamed')} is part of an articulation - skipping physics material (requires special configuration)"
                    )
                    return False

                # Check if object has physics APIs for regular rigid bodies
                has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
                has_rigidbody = prim.HasAPI(UsdPhysics.RigidBodyAPI)
                has_deformable = prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI)

                if not (has_collision or has_rigidbody or has_deformable):
                    log.debug(
                        f"Object {getattr(obj, 'name', 'unnamed')} does not have physics APIs (CollisionAPI, RigidBodyAPI, or DeformableBodyAPI) - skipping physics material"
                    )
                    return False

                log.debug(
                    f"Object {getattr(obj, 'name', 'unnamed')} has physics APIs: collision={has_collision}, rigidbody={has_rigidbody}, deformable={has_deformable}"
                )

            # Bind physics material to all environment instances
            success_count = 0
            for env_idx in range(self.sim.num_envs):
                env_prim_path = prim_path.replace("env_0", f"env_{env_idx}")
                if self._bind_material_to_prim(env_prim_path, material_path, "physics"):
                    success_count += 1

            if success_count > 0:
                log.debug(
                    f"Applied physics material '{material_name}' to {getattr(obj, 'name', 'unnamed')} (bindings: {success_count}/{self.sim.num_envs})"
                )
                return True
            else:
                log.debug(f"Physics material binding failed for {getattr(obj, 'name', 'unnamed')}")
                return False

        except Exception as e:
            log.debug(f"Failed to apply physics material to {getattr(obj, 'name', 'unnamed')}: {e}")
            return False

    def _get_object_prim_path(self, obj: Any) -> str | None:
        """Get the USD prim path for an object."""
        # Try different ways to get the prim path
        if hasattr(obj, "prim_path"):
            return str(obj.prim_path)
        elif hasattr(obj, "cfg") and hasattr(obj.cfg, "prim_path"):
            return str(obj.cfg.prim_path)
        elif hasattr(obj, "_prim_path"):
            return str(obj._prim_path)
        elif hasattr(obj, "name"):
            # Construct path from name - default to env_0 for pattern matching
            return f"/World/envs/env_0/{obj.name}"
        else:
            log.warning("Could not determine prim path for object")
            return None

    def _apply_simple_color_randomization(self, obj: Any) -> bool:
        """Apply simple color randomization as fallback."""
        try:
            if hasattr(obj, "color"):
                # Generate random color
                new_color = [
                    float(np.random.uniform(0.1, 1.0)),
                    float(np.random.uniform(0.1, 1.0)),
                    float(np.random.uniform(0.1, 1.0)),
                ]
                obj.color = new_color
                log.debug(f"Applied simple color randomization to {getattr(obj, 'name', 'unnamed')}")
                return True
        except Exception as e:
            log.warning(f"Failed to apply simple color randomization: {e}")

        return False

    def _reset_unassigned_object_materials(self, objects_with_materials: set) -> None:
        """Reset materials for objects that weren't assigned new materials to their defaults."""
        try:
            # Find objects that don't have materials assigned
            all_object_names = {obj.name for obj in self.sim.objects}
            unassigned_objects = all_object_names - objects_with_materials

            if unassigned_objects:
                log.debug(f"Resetting materials for unassigned objects: {unassigned_objects}")

                for obj_name in unassigned_objects:
                    try:
                        # Find the object configuration
                        obj_cfg = None
                        for obj in self.sim.objects:
                            if obj.name == obj_name:
                                obj_cfg = obj
                                break

                        if obj_cfg and hasattr(obj_cfg, "color"):
                            # Create a default material based on the object's original color
                            default_material_config = {
                                "type": "pbr",
                                "name": f"default_{obj_name}",
                                "diffuse_color": obj_cfg.color,
                                "metallic": 0.0,
                                "roughness": 0.5,
                                "specular": 0.2,
                                "opacity": 1.0,
                            }

                            # Apply the default material
                            self._apply_material_to_object(obj_cfg, default_material_config, "default")
                            log.debug(f"Reset material for {obj_name} to default")

                    except Exception as e:
                        log.debug(f"Failed to reset material for {obj_name}: {e}")

        except Exception as e:
            log.debug(f"Failed to reset unassigned object materials: {e}")

    def _bind_material_to_prim(self, prim_path: str, material_path: str, material_type: str = "visual") -> bool:
        """Universal material binding method for PBR, Physics, and MDL materials."""
        try:
            import omni.usd
            from pxr import UsdShade

            stage = omni.usd.get_context().get_stage()
            if not stage:
                return False

            # Verify prims exist
            target_prim = stage.GetPrimAtPath(prim_path)
            material_prim = stage.GetPrimAtPath(material_path)

            if not target_prim.IsValid() or not material_prim.IsValid():
                return False

            # Apply material binding API if needed
            if not target_prim.HasAPI(UsdShade.MaterialBindingAPI):
                UsdShade.MaterialBindingAPI.Apply(target_prim)

            mat_binding_api = UsdShade.MaterialBindingAPI(target_prim)
            material = UsdShade.Material(material_prim)

            if not material:
                return False

            if material_type == "mdl":
                try:
                    mat_binding_api.UnbindDirectBinding()
                except Exception:
                    pass
                # collect all meshes
                from pxr import Gf, Sdf, Usd, UsdGeom, Vt

                meshes = [UsdGeom.Mesh(p) for p in Usd.PrimRange(target_prim) if p.IsA(UsdGeom.Mesh)]
                # if target is a Mesh, add it to the list
                if target_prim.IsA(UsdGeom.Mesh):
                    meshes.append(UsdGeom.Mesh(target_prim))

                # fix uv for meshes without/empty uv
                def _ensure_uv(mesh, tile: float = 0.2):
                    pvapi = UsdGeom.PrimvarsAPI(mesh)
                    pv = pvapi.GetPrimvar("st")
                    vals = pv.Get() if pv else None
                    idx = mesh.GetFaceVertexIndicesAttr().Get() or []

                    if pv and vals and len(vals) > 0:
                        return  # no need to fix
                    # calculate bounding box, and project to the largest two axes

                    pts = mesh.GetPointsAttr().Get() or []
                    if not pts:
                        return
                    xs, ys, zs = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
                    rx, ry, rz = (
                        ((max(xs) - min(xs)) if xs else 0),
                        ((max(ys) - min(ys)) if ys else 0),
                        ((max(zs) - min(zs)) if zs else 0),
                    )
                    axes = sorted([("x", rx), ("y", ry), ("z", rz)], key=lambda t: t[1], reverse=True)
                    a, b = axes[0][0], axes[1][0]

                    def comp(p, axis):
                        return p[0] if axis == "x" else (p[1] if axis == "y" else p[2])

                    from math import isfinite

                    st_list = []
                    for vid in idx:
                        p = pts[vid]
                        u = comp(p, a) * tile
                        v = comp(p, b) * tile
                        u = 0.0 if not isfinite(u) else u
                        v = 0.0 if not isfinite(v) else v
                        st_list.append((u, v))

                    st = Vt.Vec2fArray([Gf.Vec2f(u, v) for (u, v) in st_list])
                    api = UsdGeom.PrimvarsAPI(mesh)
                    pv = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
                    pv.Set(st)
                    pv.SetInterpolation(UsdGeom.Tokens.faceVarying)

                for m in meshes:
                    _ensure_uv(m)

                # bind MDL to mesh, avoid being overridden by sub-meshes
                for m in meshes:
                    mprim = m.GetPrim()
                    if not mprim.HasAPI(UsdShade.MaterialBindingAPI):
                        UsdShade.MaterialBindingAPI.Apply(mprim)
                    api_m = UsdShade.MaterialBindingAPI(mprim)
                    api_m.UnbindDirectBinding()
                    api_m.Bind(material, UsdShade.Tokens.strongerThanDescendants)

            elif material_type == "physics":
                mat_binding_api.UnbindDirectBinding("physics")
                mat_binding_api.Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")
            else:
                mat_binding_api.UnbindDirectBinding()
                mat_binding_api.Bind(material, UsdShade.Tokens.strongerThanDescendants)

            return True

        except Exception as e:
            log.debug(f"Material binding failed for {prim_path}: {e}")
            return False
