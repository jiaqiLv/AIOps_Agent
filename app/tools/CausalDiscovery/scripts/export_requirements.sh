#!/bin/bash
# Export requirements.txt for the project

echo "Exporting requirements.txt..."

# Method 1: pip freeze (all packages in current environment)
echo "=== Method 1: pip freeze ==="
pip freeze > requirements_all.txt
echo "Exported all packages to requirements_all.txt"

# Method 2: pipreqs (only used packages)
echo ""
echo "=== Method 2: pipreqs (project dependencies only) ==="

# Check if pipreqs is installed
if ! command -v pipreqs &> /dev/null; then
    echo "pipreqs not found. Installing..."
    pip install pipreqs
fi

# Run pipreqs from project root (parent of this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

pipreqs --force "$PROJECT_ROOT"
echo "Exported project dependencies to requirements.txt"

echo ""
echo "Done!"
echo "  - requirements_all.txt: All packages in current environment"
echo "  - requirements.txt: Only project dependencies (recommended for deployment)"
