### set path and working directory
import sys
sys.path.append('/content/bert_for_patents/05 Analysis/01 Main')
%cd /content/bert_for_patents/05 Analysis/01 Main

### PACKAGES
import subprocess
import pkg_resources

# List of packages to skip in Colab
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

# Install only the missing packages, skip those meant for local (non-Colab)
for package in requirements:
    pkg_name = package.split('==')[0]  # Get the package name
    # Skip local-only packages
    if pkg_name.lower() in packages_to_skip:
        print(f"Skipping {pkg_name} (not needed in Colab).")
        continue
    # Check if package is installed
    if pkg_name.lower() not in installed_packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    else:
        print(f"{package} is already installed.")
