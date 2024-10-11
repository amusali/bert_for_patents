import sys
import os
import subprocess
import importlib.metadata

# Set path and working directory
sys.path.append('/content/bert_for_patents/05 Analysis/01 Main')
os.chdir("/content/bert_for_patents/05 Analysis/01 Main")

# List of packages to skip in Colab
packages_to_skip = ['tensorflow-intel', 'pywin32', 'pywin', 'win32api']

# Read the requirements.txt file and handle encoding issues
with open('/content/bert_for_patents/requirements.txt', 'rb') as f:
    content = f.read()

# Decode the file content and handle BOM
content = content.decode('utf-8-sig')  # This handles BOM if present
requirements = content.splitlines()

# Get the list of installed packages using importlib.metadata (instead of pkg_resources)
installed_packages = {pkg.metadata['Name'].lower() for pkg in importlib.metadata.distributions()}

# Install only the missing packages, skip local-only ones
for package in requirements:
    package = package.strip()
    pkg_name = package.split('==')[0].lower()  # Get the package name and make it lowercase for comparison

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

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
