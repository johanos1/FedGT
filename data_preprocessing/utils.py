import os
import yaml

def read_config(config_file):
    """Read a config file in YAML.
    Parameters
    ----------
    config_file : str
        Path towards the con fig file in YAML.
    Returns
    -------
    dict
        The parsed config
    Raises
    ------
    FileNotFoundError
        If the config file does not exist
    """
    if not (os.path.exists(config_file)):
        raise FileNotFoundError("Could not find the config to read.")
    with open(config_file, "r") as file:
        dict = yaml.load(file, Loader=yaml.FullLoader)
    return dict

def get_config_file_path(dataset_name, output_dir, debug):
    """Get the config_file path in real or debug mode.
    Parameters
    ----------
    dataset_name: str
        The name of the dataset to get the config from.
    debug : bool
       The mode in which we download the dataset.
    Returns
    -------
    str
        The path towards the config file.
    """
    assert dataset_name in [
        "fed_camelyon16",
        "fed_heart_disease",
        "fed_synthetic",
        "fed_isic2019",
        "fed_lidc_idri",
        "fed_ixi",
        "fed_kits19",
    ], f"Dataset name {dataset_name} not valid."
    config_file_name = (
        "dataset_location_debug.yaml" if debug else "dataset_location.yaml"
    )
    #datasets_dir = str(Path(os.path.realpath(datasets.__file__)).parent.resolve())
    path_to_config_file_folder = output_dir
    config_file = os.path.join(path_to_config_file_folder, config_file_name)
    return config_file


def create_config(output_folder, debug, dataset_name="fed_camelyon16"):
    """Create or modify config file by writing the absolute path of \
        output_folder in its dataset_path key.
    Parameters
    ----------
    output_folder : str
        The folder where the dataset will be downloaded.
    debug : bool
        Whether or not we are in debug mode.
    dataset_name: str
        The name of the dataset to get the config from.
    Returns
    -------
    Tuple(dict, str)
        The parsed config and the path to the file written on disk.
    Raises
    ------
    ValueError
        If output_folder is not a directory.
    """
    if not (os.path.isdir(output_folder)):
        raise ValueError(f"{output_folder} is not recognized as a folder")

    config_file = get_config_file_path(dataset_name, output_folder, debug)

    if not (os.path.exists(config_file)):
        dataset_path = os.path.realpath(output_folder)
        dict = {
            "dataset_path": dataset_path,
            "download_complete": False,
            "preprocessing_complete": False,
        }

        with open(config_file, "w") as file:
            yaml.dump(dict, file)
    else:
        dict = read_config(config_file)

    return dict, config_file


def write_value_in_config(config_file, key, value):
    """Update config_file by modifying one of its key with its new value.
    Parameters
    ----------
    config_file : str
        Path towards a config file
    key : str
        A key belonging to download_complete, preprocessing_complete, dataset_path
    value : Union[bool, str]
        The value to write for the key field.
    Raises
    ------
    ValueError
        If the config file does not exist.
    """
    if not (os.path.exists(config_file)):
        raise FileNotFoundError(
            "The config file doesn't exist. \
            Please create the config file before updating it."
        )
    dict = read_config(config_file)
    dict[key] = value
    with open(config_file, "w") as file:
        yaml.dump(dict, file)
        
    
