# ruff: noqa: F401

"""Sub-module containing the task configuration."""

import time

from loguru import logger as log

from .base_task_cfg import BaseTaskCfg
import os
import importlib


def __get_quick_ref():
    tic = time.time()

    # from .calvin.calvin import MoveSliderLeftACfg
    from .calvin.calvin import CloseDrawerACfg
    from .calvin.calvin import LiftBlueBlockDrawerACfg
    from .calvin.calvin import LiftBlueBlockSliderACfg
    from .calvin.calvin import LiftBlueBlockTableACfg
    from .calvin.calvin import LiftPinkBlockDrawerACfg
    from .calvin.calvin import LiftPinkBlockSliderACfg
    from .calvin.calvin import LiftPinkBlockTableACfg
    from .calvin.calvin import LiftRedBlockDrawerACfg
    from .calvin.calvin import LiftRedBlockSliderACfg
    from .calvin.calvin import LiftRedBlockTableACfg
    from .calvin.calvin import MoveSliderLeftACfg
    from .calvin.calvin import MoveSliderRightACfg
    from .calvin.calvin import OpenDrawerACfg
    from .calvin.calvin import PlaceInDrawerACfg
    from .calvin.calvin import PlaceInSliderACfg
    from .calvin.calvin import PushBlueBlockLeftACfg
    from .calvin.calvin import PushBlueBlockRightACfg
    from .calvin.calvin import PushIntoDrawerACfg
    from .calvin.calvin import PushPinkBlockLeftACfg
    from .calvin.calvin import PushPinkBlockRightACfg
    from .calvin.calvin import PushRedBlockLeftACfg
    from .calvin.calvin import PushRedBlockRightACfg
    from .calvin.calvin import RotateBlueBlockLeftACfg
    from .calvin.calvin import RotateBlueBlockRightACfg
    from .calvin.calvin import RotatePinkBlockLeftACfg
    from .calvin.calvin import RotatePinkBlockRightACfg
    from .calvin.calvin import RotateRedBlockLeftACfg
    from .calvin.calvin import RotateRedBlockRightACfg
    from .calvin.calvin import StackBlockACfg
    from .calvin.calvin import UnstackBlockACfg

    from .debug.reach_cfg import ReachOriginCfg
    from .dmcontrol.walker_walk_cfg import WalkerWalkCfg
    from .fetch import FetchCloseBoxCfg
    from .gapartnet import GapartnetOpenDrawerCfg
    from .humanoidbench import StandCfg
    from .isaacgym_envs.allegrohand_cfg import AllegroHandCfg
    from .isaacgym_envs.ant_isaacgym_cfg import AntIsaacGymCfg
    from .isaacgym_envs.anymal_cfg import AnymalCfg
    from .libero.libero_objects.libero_pick_alphabet_soup import LiberoPickAlphabetSoupCfg
    from .libero.libero_objects.libero_pick_bbq_sauce import LiberoPickBbqSauceCfg
    from .libero.libero_objects.libero_pick_butter import LiberoPickButterCfg
    from .libero.libero_objects.libero_pick_chocolate_pudding import LiberoPickChocolatePuddingCfg
    from .libero.libero_objects.libero_pick_cream_cheese import LiberoPickCreamCheeseCfg
    from .libero.libero_objects.libero_pick_ketchup import LiberoPickKetchupCfg
    from .libero.libero_objects.libero_pick_milk import LiberoPickMilkCfg
    from .libero.libero_objects.libero_pick_orange_juice import LiberoPickOrangeJuiceCfg
    from .libero.libero_objects.libero_pick_salad_dressing import LiberoPickSaladDressingCfg
    from .libero.libero_objects.libero_pick_tomato_sauce import LiberoPickTomatoSauceCfg
    from .maniskill.pick_cube_cfg import PickCubeCfg
    from .maniskill.pick_single_ycb import PickSingleYcbCrackerBoxCfg
    from .maniskill.stack_cube_cfg import StackCubeCfg
    from .rlafford.rl_afford_open_door_cfg import RlAffordOpenDoorCfg
    # from .rlbench.basketball_in_hoop_cfg import BasketballInHoopCfg
    # from .rlbench.close_box_cfg import CloseBoxCfg

    from .rlbench.basketball_in_hoop_cfg import BasketballInHoopCfg
    from .rlbench.beat_the_buzz_cfg import BeatTheBuzzCfg
    from .rlbench.block_pyramid_cfg import BlockPyramidCfg
    from .rlbench.change_channel_cfg import ChangeChannelCfg
    from .rlbench.change_clock_cfg import ChangeClockCfg
    from .rlbench.close_box_cfg import CloseBoxCfg
    from .rlbench.close_door_cfg import CloseDoorCfg, OpenDoorCfg
    from .rlbench.close_drawer_cfg import CloseDrawerCfg, OpenDrawerCfg
    from .rlbench.close_fridge_cfg import CloseFridgeCfg, OpenFridgeCfg
    from .rlbench.close_grill_cfg import CloseGrillCfg
    from .rlbench.close_jar_cfg import CloseJarCfg
    from .rlbench.close_laptop_lid_cfg import CloseLaptopLidCfg
    from .rlbench.close_microwave_cfg import CloseMicrowaveCfg, OpenMicrowaveCfg
    from .rlbench.empty_dishwasher_cfg import EmptyDishwasherCfg
    from .rlbench.get_ice_from_fridge_cfg import GetIceFromFridgeCfg
    from .rlbench.hang_frame_on_hanger_cfg import HangFrameOnHangerCfg
    from .rlbench.hit_ball_with_queue_cfg import HitBallWithQueueCfg
    from .rlbench.hockey_cfg import HockeyCfg
    from .rlbench.insert_onto_square_peg_cfg import InsertOntoSquarePegCfg
    from .rlbench.lamp_off_cfg import LampOffCfg, LampOnCfg
    from .rlbench.lift_numbered_block_cfg import LiftNumberedBlockCfg
    from .rlbench.light_bulb_in_cfg import LightBulbInCfg
    from .rlbench.light_bulb_out_cfg import LightBulbOutCfg
    from .rlbench.meat_off_grill_cfg import MeatOffGrillCfg, MeatOnGrillCfg
    from .rlbench.open_box_cfg import OpenBoxCfg
    from .rlbench.open_grill_cfg import OpenGrillCfg
    from .rlbench.open_jar_cfg import OpenJarCfg
    from .rlbench.open_oven_cfg import OpenOvenCfg, PutTrayInOvenCfg, TakeTrayOutOfOvenCfg
    from .rlbench.open_washing_machine_cfg import OpenWashingMachineCfg
    from .rlbench.open_window_cfg import OpenWindowCfg
    from .rlbench.open_wine_bottle_cfg import OpenWineBottleCfg
    from .rlbench.phone_on_base_cfg import PhoneOnBaseCfg
    from .rlbench.pick_and_lift_cfg import PickAndLiftCfg
    from .rlbench.pick_and_lift_small_cfg import PickAndLiftSmallCfg
    from .rlbench.pick_up_cup_cfg import PickUpCupCfg
    from .rlbench.place_cups_cfg import PlaceCupsCfg, RemoveCupsCfg
    from .rlbench.place_shape_in_shape_sorter_cfg import PlaceShapeInShapeSorterCfg
    from .rlbench.play_jenga_cfg import PlayJengaCfg
    from .rlbench.plug_charger_in_power_supply_cfg import PlugChargerInPowerSupplyCfg
    from .rlbench.pour_from_cup_to_cup_cfg import PourFromCupToCupCfg
    from .rlbench.press_switch_cfg import PressSwitchCfg
    from .rlbench.push_button_cfg import PushButtonCfg
    from .rlbench.push_buttons_cfg import PushButtonsCfg
    from .rlbench.put_books_on_bookshelf_cfg import PutBooksOnBookshelfCfg
    from .rlbench.put_bottle_in_fridge_cfg import PutBottleInFridgeCfg
    from .rlbench.put_groceries_in_cupboard_cfg import PutAllGroceriesInCupboardCfg, PutGroceriesInCupboardCfg
    from .rlbench.put_item_in_drawer_cfg import PutItemInDrawerCfg
    from .rlbench.put_knife_in_knife_block_cfg import PutKnifeInKnifeBlockCfg, PutKnifeOnChoppingBoardCfg
    from .rlbench.put_money_in_safe_cfg import PutMoneyInSafeCfg
    from .rlbench.put_plate_in_colored_dish_rack_cfg import PutPlateInColoredDishRackCfg
    from .rlbench.put_rubbish_in_bin_cfg import PutRubbishInBinCfg
    from .rlbench.put_shoes_in_box_cfg import PutShoesInBoxCfg
    from .rlbench.put_toilet_roll_on_stand_cfg import PutToiletRollOnStandCfg
    from .rlbench.put_umbrella_in_umbrella_stand_cfg import PutUmbrellaInUmbrellaStandCfg
    from .rlbench.reach_and_drag_cfg import ReachAndDragCfg
    from .rlbench.reach_target_cfg import ReachTargetCfg
    from .rlbench.scoop_with_spatula_cfg import ScoopWithSpatulaCfg
    from .rlbench.screw_nail_cfg import ScrewNailCfg
    from .rlbench.set_the_table_cfg import SetTheTableCfg
    from .rlbench.setup_checkers_cfg import SetupCheckersCfg
    from .rlbench.setup_chess_cfg import SetupChessCfg
    from .rlbench.slide_block_to_target_cfg import SlideBlockToTargetCfg
    from .rlbench.slide_cabinet_open_and_place_cups_cfg import SlideCabinetOpenAndPlaceCupsCfg, TakeCupOutFromCabinetCfg
    from .rlbench.stack_blocks_cfg import StackBlocksCfg
    from .rlbench.stack_chairs_cfg import StackChairsCfg
    from .rlbench.stack_cups_cfg import StackCupsCfg
    from .rlbench.stack_wine_cfg import StackWineCfg
    from .rlbench.sweep_to_dustpan_cfg import SweepToDustpanCfg

    # from .rlbench import *
    from .robosuite import SquareD0Cfg, SquareD1Cfg, SquareD2Cfg, StackD0Cfg
    from .simpler_env.simpler_env_grasp_opened_coke_can_cfg import SimplerEnvGraspOpenedCokeCanCfg
    from .simpler_env.simpler_env_move_near import SimplerEnvMoveNearCfg

    # from .skillblender import G1BaseTaskCfg, H1BaseTaskCfg
    from .uh1 import MabaoguoCfg

    toc = time.time()

    log.trace(f"Time taken to load quick ref: {toc - tic:.2f} seconds")

    return locals()


__quick_ref = __get_quick_ref()


def __getattr__(name):
    if name in __quick_ref:
        return __quick_ref[name]

    if name.startswith("GraspNet") and name.endswith("Cfg"):
        from .graspnet import __getattr__ as graspnet_getattr

        return graspnet_getattr(name)

    elif name.startswith("GAPartManip") and name.endswith("Cfg"):
        from .gapartmanip import __getattr__ as gapartmanip_getattr

        return gapartmanip_getattr(name)

    else:
        raise AttributeError(f"Module {__name__} has no attribute {name}")
        raise AttributeError(f"Module {__name__} has no attribute {name}")
