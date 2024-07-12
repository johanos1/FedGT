from __future__ import division
import os
import sys
import zipfile
import glob
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

from data_preprocessing.utils import create_config, write_value_in_config, read_config

def color_constancy(img, power=6, gamma=None):
    """
    Preprocessing step to make sure that the images appear with similar brightness
    and contrast.
    See this [link}(https://en.wikipedia.org/wiki/Color_constancy) for an explanation.
    Thank you to [Aman Arora](https://github.com/amaarora) for this
    [implementation](https://github.com/amaarora/melonama)
    Parameters
    ----------
    img: 3D numpy array, the original image
    power: int, degree of norm
    gamma: float, value of gamma correction
    """
    img_dtype = img.dtype

    # if gamma is not None:
    #     img = img.astype("uint8")
    #     look_up_table = np.ones((256, 1), dtype="uint8") * 0
    #     for i in range(256):
    #         look_up_table[i][0] = 255 * pow(i / 255, 1 / gamma)
    #     img = cv2.LUT(img, look_up_table)

    if gamma is not None:
        img = img.astype("uint8")
        gamma_inv = 1.0 / gamma
        img = np.clip(255 * np.power(img / 255.0, gamma_inv), 0, 255).astype("uint8")

    img = img.astype("float32")
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    return img.astype(img_dtype)

def resize_images(output_dir):
    dir = output_dir
    config_file = os.path.join(dir, "dataset_location.yaml")
    dict = read_config(config_file)
    if not (dict["download_complete"]):
        raise ValueError("Download incomplete. Please relaunch the download script")
    if dict["preprocessing_complete"]:
        print("You have already ran the preprocessing, aborting.")
        sys.exit()
    input_path = dict["dataset_path"]


    dic = {
        "inputs": "ISIC_2019_Training_Input",
        "inputs_preprocessed": "ISIC_2019_Training_Input_preprocessed",
    }
    input_folder = os.path.join(input_path, dic["inputs"])
    output_folder = os.path.join(input_path, dic["inputs_preprocessed"])
    os.makedirs(output_folder, exist_ok=True)


def resize_and_maintain(path, output_path, sz: tuple, cc):
    """Preprocessing of images
    Mantains aspect ratio fo input image. Possibility to add color constancy.
    Thank you to [Aman Arora](https://github.com/amaarora) for this
    [implementation](https://github.com/amaarora/melonama)
    Parameters
    ----------
    path : path to input image
    output_path : path to output image
    sz : tuple, shorter edge of resized image is sz[0]
    cc : color constancy is added if True
    """
    fn = os.path.basename(path)
    img = Image.open(path)
    size = sz[0]
    old_size = img.size
    ratio = float(size) / min(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, resample=Image.BILINEAR)
    if cc:
        img = color_constancy(np.array(img))
        img = Image.fromarray(img)
    img.save(os.path.join(output_path, fn))
   

def download_isic():
    url_1 = (
        "https://isic-challenge-data.s3.amazonaws.com/2019/" "ISIC_2019_Training_Input.zip"
    )

    url_2 = (
        "https://isic-challenge-data.s3.amazonaws.com/2019/"
        "ISIC_2019_Training_Metadata.csv"
    )

    url_3 = (
        "https://isic-challenge-data.s3.amazonaws.com/2019/"
        "ISIC_2019_Training_GroundTruth.csv"
    )

    # Creating output folder
    current_directory = os.getcwd()  # Get current directory
    output_folder = os.path.join(current_directory, "data/fed_isic2019")
    os.makedirs(output_folder, exist_ok=True)

    # Creating config file with path to dataset from arguments
    dict, config_file = create_config(
        output_folder=output_folder,
        debug=False, dataset_name="fed_isic2019"
    )
    if dict["download_complete"]:
        print("You already have downloaded the dataset. Aborting.")
    else:
        data_directory = dict["dataset_path"]

        
        dest_file_1 = os.path.join(data_directory, "ISIC_2019_Training_Input.zip")
        dest_file_2 = os.path.join(data_directory, "ISIC_2019_Training_Metadata.csv")
        dest_file_3 = os.path.join(data_directory, "ISIC_2019_Training_GroundTruth.csv")
        dest_file_4 = os.path.join(data_directory, "ISIC_2019_Training_Metadata_FL.csv")
        file1 = os.path.join(current_directory, "ISIC2019/HAM10000_metadata")

        # download and unzip data
        os.system(f"wget {url_1} --no-check-certificate -O {dest_file_1}")
        if zipfile.is_zipfile(dest_file_1):
            print("Zip file downloaded correctly")
            os.system(f"unzip {dest_file_1} -d {data_directory}")
            os.system(f"rm {dest_file_1}")
        else:
            sys.exit("Zip file corrupted")
        os.system(f"wget {url_2} --no-check-certificate -O {dest_file_2}")
        os.system(f"wget {url_3} --no-check-certificate -O {dest_file_3}")

        # create pandas dataframes
        ISIC_2019_Training_Metadata = pd.read_csv(dest_file_2)
        ISIC_2019_Training_GroundTruth = pd.read_csv(dest_file_3)
        # keeping only image and dataset columns in the HAM10000 metadata
        HAM10000_metadata = pd.read_csv(file1)
        HAM10000_metadata.rename(columns={"image_id": "image"}, inplace=True)
        HAM10000_metadata.drop(
            ["age", "sex", "localization", "lesion_id", "dx", "dx_type"], axis=1, inplace=True
        )

        # taking out images (from image set, metadata file and ground truth file)
        # where datacenter is not available
        for i, row in ISIC_2019_Training_Metadata.iterrows():
            if pd.isnull(row["lesion_id"]):
                image = row["image"]
                os.system("rm " + data_directory + "/ISIC_2019_Training_Input/" + image + ".jpg")
                if image != ISIC_2019_Training_GroundTruth["image"][i]:
                    print("Mismatch between Metadata and Ground Truth")
                ISIC_2019_Training_GroundTruth = ISIC_2019_Training_GroundTruth.drop(i)
                ISIC_2019_Training_Metadata = ISIC_2019_Training_Metadata.drop(i)

        # generating dataset field from lesion_id field in the metadata dataframe
        ISIC_2019_Training_Metadata["dataset"] = ISIC_2019_Training_Metadata["lesion_id"].str[:4]

        # join with HAM10000 metadata in order to expand the HAM datacenters
        result = pd.merge(ISIC_2019_Training_Metadata, HAM10000_metadata, how="left", on="image")
        result["dataset"] = result["dataset_x"] + result["dataset_y"].astype(str)
        result.drop(["dataset_x", "dataset_y", "lesion_id"], axis=1, inplace=True)

        # checking sizes and saving to csv files
        print("Datacenters")
        print(result["dataset"].value_counts())
        print("Number of lines in Metadata", ISIC_2019_Training_Metadata.shape[0])
        print("Number of lines in GroundTruth", ISIC_2019_Training_GroundTruth.shape[0])
        print("Number of lines in MetadataFL", result.shape[0])
        DIR = os.path.join(data_directory, "ISIC_2019_Training_Input")
        N = (
            len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
            - 2
        )
        print("Number of images", N)
        result.to_csv(dest_file_4, index=False)
        ISIC_2019_Training_Metadata.to_csv(dest_file_2, index=False)
        ISIC_2019_Training_GroundTruth.to_csv(dest_file_3, index=False)

        if N == 23247:
            print("Download OK")
            write_value_in_config(config_file, "download_complete", True)
        else:
            print("Something wrong happened during the download.")

    if not dict["preprocessing_complete"]:
        # resize images
        data_folder = os.path.join(output_folder, "ISIC_2019_Training_Input")
        preprocessed_folder = os.path.join(output_folder, "ISIC_2019_Training_Input_preprocessed")
        os.makedirs(preprocessed_folder, exist_ok=True)
        images = glob.glob(os.path.join(data_folder, "*.jpg"))
        cc = True
        sz = 224
        
        print(
            "Resizing images to mantain aspect ratio in a way that the shorter side"
            " is {}px but images are rectangular.".format(sz)
        )
        # resize and adjust constrast in images
        Parallel(n_jobs=32)(
            delayed(resize_and_maintain)(i, preprocessed_folder, (sz, sz), cc)
            for i in tqdm(images)
        )
        write_value_in_config(config_file, "preprocessing_complete", True)
    else:
        data_directory = dict["dataset_path"]
        dest_file_3 = os.path.join(data_directory, "ISIC_2019_Training_GroundTruth.csv")
        dest_file_4 = os.path.join(data_directory, "ISIC_2019_Training_Metadata_FL.csv")
        preprocessed_folder = os.path.join(output_folder, "ISIC_2019_Training_Input_preprocessed")
        ISIC_2019_Training_GroundTruth = pd.read_csv(dest_file_3)
        result = pd.read_csv(dest_file_4)
        print("Dataset already preprocessed.")
        
    return ISIC_2019_Training_GroundTruth, result, preprocessed_folder
