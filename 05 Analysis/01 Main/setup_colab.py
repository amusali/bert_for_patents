import sys
import os
import subprocess
import importlib.metadata

# Set path and working directory
sys.path.append('/content/bert_for_patents/05 Analysis/01 Main')
os.chdir("/content/bert_for_patents/05 Analysis/01 Main")
import subprocess
import pkg_resources

packages_to_skip = ['tensorflow-intel', 'pywin32', 'pywin', 'win32api']

# Read the requirements.txt file in binary mode
with open('/content/bert_for_patents/requirements.txt', 'rb') as f:
    content = f.read()

# Remove BOM if present
content = content.decode('utf-8-sig')  # 'utf-8-sig' handles BOM

# Split lines and store in requirements
requirements = content.splitlines()

# Get the list of installed packages
installed_packages = {pkg.key for pkg in pkg_resources.working_set}

# Install only the missing packages
for package in requirements:
    pkg_name = package.split('==')[0]  # Get the package name

    if pkg_name.lower() in packages_to_skip:
        print(f"Skipping {pkg_name} (not needed in Colab).")
        continue
    if pkg_name not in installed_packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    else:
        print(f"{package} is already installed.")
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
