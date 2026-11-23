
import os
import sys
import subprocess
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# Paths
BASE_DIR = Path("/content/bert_for_patents")
WORK_DIR = BASE_DIR / "05 Analysis/01 Main"
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"

# Set working directory and Python path
os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR))

# Packages not needed in Colab
PACKAGES_TO_SKIP = {
    "tensorflow-intel",
    "pywin32",
    "pywin",
    "win32api",
}

# Read requirements (UTF-16 LE)
requirements = (
    REQUIREMENTS_FILE.read_text(encoding="utf-16-le")
    .lstrip("\ufeff")
    .splitlines()
)

# Install missing packages
for line in requirements:
    line = line.strip()

    # Skip empty lines and comments
    if not line or line.startswith("#"):
        continue

    # Remove inline comments
    line = line.split(" #", 1)[0].strip()

    # Parse package requirement
    req = Requirement(line)
    pkg_name = canonicalize_name(req.name)

    # Skip Windows-specific packages
    if pkg_name in PACKAGES_TO_SKIP:
        print(f"Skipping {pkg_name} (not needed in Colab).")
        continue

    # Skip requirements not applicable to this environment
    if req.marker and not req.marker.evaluate():
        print(f"Skipping {line} (environment mismatch).")
        continue

    # Check whether already installed
    try:
        installed_version = version(req.name)
        print(f"{pkg_name}=={installed_version} is already installed.")
        continue
    except PackageNotFoundError:
        pass

    # Install missing package
    print(f"Installing {line}...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", line
    ])

print("\nDependency setup completed.")