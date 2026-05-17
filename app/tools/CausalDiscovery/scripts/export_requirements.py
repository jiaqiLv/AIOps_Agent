#!/usr/bin/env python3
"""
Export requirements.txt for the project.

This script provides two methods:
1. pip freeze - All packages in current environment
2. pipreqs - Only project dependencies (by analyzing imports)
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n=== {description} ===")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(e.stderr)
        return False


def main():
    """Main function."""
    print("Exporting requirements.txt...")

    # Get project root (parent of scripts directory)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Method 1: pip freeze
    all_req_path = project_root / "requirements_all.txt"
    if run_command(
        f"pip freeze > \"{all_req_path}\"",
        "Method 1: pip freeze (all packages)"
    ):
        print(f"Exported all packages to: {all_req_path}")

    # Method 2: pipreqs
    print("\n=== Method 2: pipreqs (project dependencies only) ===")

    # Check if pipreqs is installed
    check_result = subprocess.run(
        ["pip", "show", "pipreqs"],
        capture_output=True
    )

    if check_result.returncode != 0:
        print("pipreqs not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pipreqs"], check=True)

    # Run pipreqs
    req_path = project_root / "requirements.txt"
    subprocess.run(
        ["pipreqs", "--force", str(project_root)],
        check=True
    )
    print(f"Exported project dependencies to: {req_path}")

    print("\nDone!")
    print(f"  - {all_req_path}: All packages in current environment")
    print(f"  - {req_path}: Only project dependencies (recommended for deployment)")


if __name__ == "__main__":
    main()