### set path and working directory
import sys
import os 

sys.path.append('/content/bert_for_patents/05 Analysis/01 Main')
# Change directory to the location of the setup_colab.py file
os.chdir("/content/bert_for_patents/05 Analysis/01 Main")

### PACKAGES
import subprocess
import pkg_resources

# List of packages to skip in Colab
packages_to_skip = ['tensorflow-intel', 'pywin32', 'pywin', 'win32api']

# Read the requirements.txt file
with open('/content/bert_for_patents/requirements.txt', 'r') as f:
    requirements = f.readlines()

# Get the list of installed packages
installed_packages = {pkg.key for pkg in pkg_resources.working_set}

# Install only the missing packages
for package in requirements:
    package = package.strip()
    pkg_name = package.split('==')[0]  # Get the package name

    # Skip local-only packages
    if pkg_name.lower() in packages_to_skip:
        print(f"Skipping {pkg_name} (not needed in Colab).")
        continue
    
    if pkg_name not in installed_packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    else:
        print(f"{package} is already installed.")


### MOUNT DRIVE
from google.colab import drive
drive.mount('/content/drive')