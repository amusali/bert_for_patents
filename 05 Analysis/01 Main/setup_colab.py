import sys
import os
import subprocess
import importlib.metadata

# Set path and working directory
sys.path.append('/content/bert_for_patents/05 Analysis/01 Main')
os.chdir("/content/bert_for_patents/05 Analysis/01 Main")

# List of packages to skip in Colab
packages_to_skip = ['tensorflow-intel', 'pywin32', 'pywin', 'win32api']

# Read the requirements.txt file
try:
    with open('/content/bert_for_patents/requirements.txt', 'r') as f:
        requirements = f.readlines()

    # Get the list of installed packages using importlib.metadata
    installed_packages = {pkg.metadata['Name'].lower() for pkg in importlib.metadata.distributions()}

    # Process and install only valid packages
    for package in requirements:
        package = package.strip()

        # Skip blank lines or comments
        if not package or package.startswith("#"):
            continue

        # Check if it's in the expected format "package==version"
        if "==" not in package:
            print(f"Skipping invalid package line: {package}")
            continue

        pkg_name = package.split('==')[0].lower()

        # Skip local-only packages
        if pkg_name in packages_to_skip:
            print(f"Skipping {pkg_name} (not needed in Colab).")
            continue

        # Install missing packages
        if pkg_name not in installed_packages:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            print(f"{package} is already installed.")

except Exception as e:
    print(f"Error while reading or processing the requirements file: {e}")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
