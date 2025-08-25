#!/usr/bin/env python3
"""
Debug script to check contact forces and termination contact indices mapping
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from humanoid_visualrl.wrapper.walking_wrapper import WalkingWrapper
from metasim.scenario.scenario import ScenarioCfg
from humanoid_visualrl.cfg.humanoidVisualRLCfg import BaseTableHumanoidTaskCfg


def debug_contact_forces():
    """Debug contact forces and termination contact indices mapping"""

    # Create a minimal scenario configuration
    scenario = ScenarioCfg()
    scenario.robots = ["g1"]

    try:
        # Create the wrapper
        wrapper = WalkingWrapper(scenario)

        print("=== Debug Information ===")
        print(f"Robot name: {wrapper.robot.name}")

        # Get body names from the environment
        body_names = wrapper.env.get_body_names(wrapper.robot.name)
        print(f"\nBody names (sorted): {body_names}")

        # Get body names without sorting
        body_names_unsorted = wrapper.env.get_body_names(wrapper.robot.name, sort=False)
        print(f"Body names (unsorted): {body_names_unsorted}")

        # Check termination contact indices
        print(f"\nTermination contact indices: {wrapper.termination_contact_indices}")
        print(f"Termination contact indices shape: {wrapper.termination_contact_indices.shape}")

        # Check contact forces shape
        print(f"\nContact forces shape: {wrapper.contact_forces.shape}")

        # Map indices to body names
        if len(wrapper.termination_contact_indices) > 0:
            print("\nTermination contact body names:")
            for idx in wrapper.termination_contact_indices:
                if idx < len(body_names):
                    print(f"  Index {idx}: {body_names[idx]}")
                else:
                    print(f"  Index {idx}: OUT OF RANGE!")

        # Check the robot configuration
        print(f"\nRobot terminate_contacts_links: {wrapper.robot.terminate_contacts_links}")

        # Check if there's a mismatch between expected and actual body names
        expected_links = wrapper.robot.terminate_contacts_links
        print(f"\nChecking expected links against actual body names:")
        for link in expected_links:
            matches = [i for i, name in enumerate(body_names) if link in name]
            print(f"  '{link}' -> matches: {matches}")
            if not matches:
                print(f"    WARNING: No matches found for '{link}'")

    except Exception as e:
        print(f"Error creating wrapper: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_contact_forces()
