#!/usr/bin/env python3
"""
Debug script to check G1 robot configuration and body names
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def debug_g1_config():
    """Debug G1 robot configuration"""

    try:
        # Import the G1 configuration
        from roboverse_pack.robots.g1_23dof_cfg import G1Cfg

        print("=== G1 Robot Configuration ===")

        # Create a G1 config instance
        g1_config = G1Cfg()

        print(f"Robot name: {g1_config.name}")
        print(f"Terminate contacts links: {g1_config.terminate_contacts_links}")
        print(f"Penalized contacts links: {g1_config.penalized_contacts_links}")
        print(f"Feet links: {g1_config.feet_links}")
        print(f"Knee links: {g1_config.knee_links}")
        print(f"Torso links: {g1_config.torso_links}")

        # Check if the terminate_contacts_links are properly defined
        print(f"\n=== Analysis ===")
        print(f"Number of terminate contact links: {len(g1_config.terminate_contacts_links)}")

        # Check for potential issues
        for link in g1_config.terminate_contacts_links:
            print(f"  '{link}' - length: {len(link)}")

        # Check if there are any empty or problematic strings
        empty_links = [link for link in g1_config.terminate_contacts_links if not link.strip()]
        if empty_links:
            print(f"WARNING: Found empty links: {empty_links}")

        # Check for special characters or spaces
        problematic_links = [link for link in g1_config.terminate_contacts_links if " " in link or "\t" in link]
        if problematic_links:
            print(f"WARNING: Found links with spaces/tabs: {problematic_links}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_g1_config()
