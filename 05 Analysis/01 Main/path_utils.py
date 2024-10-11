import os
import json

def get_base_path():
    """
    Determines the base path based on the environment (Colab or local).
    Returns the base path defined in config.json.
    """
    # Check if running in Colab or locally
    if 'COLAB_GPU' in os.environ:
        current_directory = os.getcwd()  # In Colab
    else:
        current_directory = os.path.dirname(os.path.abspath(__file__))  # Locally

    # Construct the path to config.json
    config_file_path = os.path.join(current_directory, 'config.json')

    # Load the config.json file
    try:
        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {config_file_path}")

    # Return the appropriate base path
    if 'COLAB_GPU' in os.environ:
        return config['colab']['base_path']
    else:
        return config['local']['base_path']
