import numpy as np
from datetime import datetime, timedelta
import itertools
import pandas as pd
from zipfile import ZipFile
import os
import shutil
import yaml
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
import urllib.request
import pickle
from typing import Union
import numbers
import itertools
import re
import math
from sklearn.model_selection import PredefinedSplit


def get_sum(a, b):
    """Demo function for the library"""
    return a + b


def get_outliers(data):
    """Lấy các giá trị outlier nằm ngoài khoảng Q1 - 1.5*IQR và Q3 + 1.5*IQR
    Args:
        data (_type_): một mảng các số

    Returns:
        _type_: các số outliers
    """

    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data < lower_bound) | (data > upper_bound)]

    return outliers


@ensure_annotations
def get_exact_day(seconds_since_epoch: int):
    """Get the exact day from 1/1/1970

    Args:
        seconds_since_epoch (int): seconds

    Returns:
        datetime: the exact day
    """

    epoch = datetime(1970, 1, 1)
    return epoch + timedelta(seconds=seconds_since_epoch)


@ensure_annotations
def is_number(string_to_check: str):
    """Check if string_to_check is a number"""

    try:
        float(string_to_check)
        return True
    except ValueError:
        return False


@ensure_annotations
def is_integer_str(s: str):
    """Check if str is an integer

    Args:
        s (str): _description_
    """

    regex = "^[+-]?\d+$"
    return re.match(regex, s) is not None


@ensure_annotations
def is_natural_number_str(s: str):
    """Check if str is a natural_number

    Args:
        s (str): _description_
    """

    regex = "^\+?\d+$"
    return re.match(regex, s) is not None


@ensure_annotations
def is_natural_number(num: numbers.Real):
    """Check if num is a natural number

    Args:
        num (numbers.Real): _description_

    """

    return is_integer(num) and num >= 0


@ensure_annotations
def is_integer(number: numbers.Real):
    """Check if number is a integer

    Args:
        number (numbers.Real): _description_
    """

    return pd.isnull(number) == False and number == int(number)


def get_combinations_k_of_n(arr, k):
    """Get the combinations k of arr having n elements"""

    return list(itertools.combinations(arr, k))


def show_frequency_table(arr):
    """Show the frequency table of arr"""
    counts, bin_edges = np.histogram(arr, bins="auto")

    frequency_table = pd.DataFrame(
        {"Bin Start": bin_edges[:-1], "Bin End": bin_edges[1:], "Count": counts}
    )

    frequency_table["Percent"] = frequency_table["Count"] / len(arr) * 100
    frequency_table["Percent"] = frequency_table["Percent"].round(2)

    return frequency_table


def extract_zip_file(zip_file_path, unzip_path):
    """Extract zip file

    Args:
        zip_file_path (str): file path of zip file
        unzip_path (str): folder path of unzip components
    """

    with ZipFile(zip_file_path, "r") as zip:
        zip.extractall(path=unzip_path)


def create_sub_folder_from_dataset(path, data_proportion, root_dir):
    """
    Args:
        path: the path to the folder to create
        root_dir : the path to the folder Dataset
        data_proportion: the proportion of taken data
    Returns:
        _str_: result
    Examples:
        vd tạo thư mục train là 70% dữ liệu, thư mục val là 15% dữ liệu, thư mục test là 15% dữ liệu

        lưu ý là di chuyển các ảnh chứ không phải copy

        nên tập val lấy 0.5 = 0.5 đối với dữ liệu còn lại

        Code:
        ```python
        dataset_path = './Dataset'

        path = './train'
        create_sub_folder_from_dataset(path, 0.7, dataset_path)

        path = './val'
        create_sub_folder_from_dataset(path, 0.5, dataset_path)

        path = './test'
        create_sub_folder_from_dataset(path, 1, dataset_path)
        ```
    """

    if not os.path.exists(path):
        os.mkdir(path)

        for dir in os.listdir(root_dir):
            os.makedirs(os.path.join(path, dir))

            img_names = np.random.permutation(os.listdir(os.path.join(root_dir, dir)))
            count_selected_img_names = int(data_proportion * len(img_names))
            selected_img_names = img_names[:count_selected_img_names]

            for img in selected_img_names:
                src = os.path.join(root_dir, dir, img)
                dest = os.path.join(path, dir)
                shutil.move(src, dest)

        return "Create the sub-folder successfully"
    else:
        return "The sub-folder existed"


def fetch_source_url_to_zip_file(source_url, local_zip_path):
    """Download file from url to local

    Returns:
        filename: the name of file
        headers: info about the file
    """

    os.makedirs(local_zip_path, exist_ok=True)
    filename, headers = urllib.request.urlretrieve(source_url, local_zip_path)

    return filename, headers


@ensure_annotations
def split_numpy_array(
    data: np.ndarray, ratios: list, dimension=1, shuffle: bool = True
):
    """

    Args:
        data (np.ndarray): _description_
        ratios (list): Tỉ lệ các phần. Tổng phải bằng 1
        dimension (int, optional): Chiều của dữ liệu. nếu dữ liệu 2 chiều thì gán = 2. Defaults to 1.
        shuffle(bool, optional): có xáo trộn dữ liệu trước khi chia không

    Returns:
        list: list các mảng numpy

    vd:
    với dữ liệu 2 chiều:
    ```python
    split_ratios = [0.5, 0.2, 0.2, 0.1]  # Tỷ lệ mong muốn
    subsets = split_data(data, split_ratios, 2)
    ```
    """
    if sum(ratios) != 1:
        raise ValueError("Tổng của ratios phải bằng 1")

    if shuffle:
        data = np.random.permutation(data)

    len_data = len(data) if dimension == 1 else data.shape[0]
    split_indices = np.cumsum(ratios)[:-1] * len_data
    split_indices = split_indices.astype(int)
    return (
        np.split(data, split_indices)
        if dimension == 1
        else np.split(data, split_indices, axis=0)
    )


@ensure_annotations
def split_dataframe_data(data: pd.DataFrame, ratios: list, shuffle: bool = True):
    """

    Args:
        data (pd.DataFrame): _description_
        ratios (list): Tỉ lệ các phần. Tổng phải bằng 1
        shuffle(bool, optional): có xáo trộn dữ liệu trước khi chia không. Defaults to True

    Returns:
        list: list các dataframe

    VD:
        ```python
    split_ratios = [0.5, 0.2, 0.2, 0.1]  # Tỷ lệ mong muốn
    subsets = split_data(data, split_ratios)
    """
    if sum(ratios) != 1:
        raise ValueError("Tổng của ratios phải bằng 1")

    if shuffle:
        data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)

    split_indices = np.cumsum(ratios)[:-1] * len(data)
    split_indices = split_indices.astype(int)

    subsets = np.split(data, split_indices, axis=0)

    return [pd.DataFrame(item, columns=data.columns) for item in subsets]


def load_python_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise e


def save_python_object(file_path, obj):
    """Save python object in a file

    Args:
        file_path (_type_): ends with .pkl
    """

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise e


def np_arange_int(start, end, step):
    """Tạo ra dãy số nguyên cách nhau step"""

    return np.arange(start, end + step, step)


def np_arange_float(start, end, step):
    """Tạo ra dãy số thực cách nhau step"""

    return np.arange(start, end, step)


def np_arange(start, end, step):
    """Create numbers from start to end with step, used for both **float** and **int** number <br>
    used for int: start, end, step must be int <br>
    used for float: start must be float <br>
    """

    if is_integer(start) and is_integer(end) and is_integer(step):
        return np_arange_int(int(start), int(end), int(step))

    return np_arange_float(start, end, step)


def get_range_for_param(param_str):
    """Create values range from param_str

    VD:
        param_str = format=start-end-step 12-15-1
        param_str = format=num 12
        param_str = format=start-end start, mean, end vd: 12-15 -> 12 13 15



    """
    if "-" not in param_str:
        if is_integer_str(param_str):
            return [int(param_str)]

        return [float(param_str)]

    if param_str.count("-") == 2:
        nums = param_str.split("-")
        num_min = float(nums[0])
        num_max = float(nums[1])
        num_step = float(nums[2])

        return np_arange(num_min, num_max, num_step)

    nums = param_str.split("-")
    num_min = float(nums[0])
    num_max = float(nums[1])

    num_mean = None
    if is_integer(num_min) and is_integer(num_max):
        num_min = int(num_min)
        num_max = int(num_max)

        num_mean = int((num_min + num_max) / 2)
    else:
        num_mean = (num_min + num_max) / 2

    return [num_min, num_mean, num_max]


@ensure_annotations
def generate_grid_search_params(param_grid: dict):
    """Generate all combinations of params like grid search

    Returns:
        list:
    """

    keys = param_grid.keys()
    values = (param_grid[key] for key in keys)
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def get_num_de_cac_combinations(list_of_list):
    """Count the number of De cac combinations of list_of_list, which is the list of list"""

    return math.prod(map(len, list_of_list))


def get_features_target_spliter_for_CV_train_val(
    train_features, train_target, val_features, val_target
):
    """Get total features, target, spliter to do GridSearchCV or RandomisedSearchCV with type is **train-val**

    Args:
        train_features (dataframe): _description_
        train_target (dataframe): _description_
        val_features (dataframe): _description_
        val_target (dataframe): _description_
    """

    features = pd.concat([train_features, val_features], axis=0)
    target = pd.concat([train_target, val_target], axis=0)
    spliter = PredefinedSplit(
        test_fold=[-1] * len(train_features) + [0] * len(val_features)
    )

    return features, target, spliter


def get_features_target_spliter_for_CV_train_train(train_features, train_target):
    """Get total features, target, spliter to do GridSearchCV or RandomisedSearchCV with type is **train-train** <br>
    When you want to train on training set and assess on that training set


    Args:
        train_features (dataframe): _description_
        train_target (dataframe): _description_
        val_features (dataframe): _description_
        val_target (dataframe): _description_
    """

    features = pd.concat([train_features, train_features], axis=0)
    target = pd.concat([train_target, train_target], axis=0)
    spliter = PredefinedSplit(
        test_fold=[-1] * len(train_features) + [0] * len(train_features)
    )

    return features, target, spliter
