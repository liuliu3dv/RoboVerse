from pxr import Usd

stage = Usd.Stage.Open(
    "/home/panwei/RoboVerse/roboverse_data/robots/g1/xml/g1_29dof_lock_waist_rev_1_0_modified_lower_fixed.usd"
)


def dump(prim, indent=0):
    print("  " * indent + prim.GetName(), prim.GetTypeName())
    for child in prim.GetChildren():
        dump(child, indent + 1)


dump(stage.GetPseudoRoot())
