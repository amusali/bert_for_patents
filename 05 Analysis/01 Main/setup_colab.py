### set path and working directory
import sys
import os 

sys.path.append('/content/bert_for_patents/05 Analysis/01 Main')
# Change directory to the location of the setup_colab.py file
os.chdir("/content/bert_for_patents/05 Analysis/01 Main")


### PACKAGES
import subprocess
import importlib.util

# Function to check if a package is installed
def package_is_installed(package_name):
  """
  Check if a package is installed and importable.
  """
  spec = importlib.util.find_spec(package_name)
  return spec is not None

# List of packages to skip in Colab
packages_to_skip = ['tensorflow-intel', 'pywin32']

requirements_path = '/content/bert_for_patents/requirements.txt'

with open(requirements_path, 'r') as file:
  requirements = [line.strip() for line in file.splitlines() if line.strip()]

for package in requirements:
  package_name = package.split(';')[0].split('==')[0].split('>=')[0].split('~')[0].strip()
  if package_name not in packages_to_skip and not package_is_installed(package_name):
    try:
      # This will install any version of the package
      subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError:
      print(f"Error installing {package}")
      raise


### MOUNT DRIVE
from google.colab import drive
drive.mount('/content/drive')