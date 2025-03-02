import os
from box.exceptions import BoxValueError
import yaml
from classifier import (
    logger,
)  # sửa <classifier> thành tên source code project tương ứng
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import pickle
import pandas as pd
import plotly.express as px
from classifier.Mylib import myfuncs
import pandas as pd
import os
from classifier import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from classifier.Mylib import myfuncs
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from xgboost import XGBClassifier
from scipy.stats import randint
import random
from lightgbm import LGBMClassifier
from sklearn.model_selection import ParameterSampler
from sklearn import metrics
from sklearn.base import clone
from classifier.utils import common


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (Path): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: _description_
    """
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        verbose (bool, optional): ignore if multiple dirs is to be created. Defaults to True.

    """

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as f:
        f.write(imgdata)


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())


def create_param_grid_from_param_grid_for_transformer_and_for_model(
    param_transformer: dict, param_model: dict
):
    a = param_transformer
    b = param_model

    c_key = []
    c_value = []
    if a is not None:
        a_key = ["transform__" + item for item in a.keys()]
        c_key = a_key
        c_value = list(a.values())

    b_key = ["model__" + item for item in b.keys()]

    c_key += b_key
    c_value += list(b.values())

    return dict(zip(c_key, c_value))


def unindent_all_lines(content):
    content = content.strip("\n")
    lines = content.split("\n")
    lines = [item.strip() for item in lines]
    content_processed = "\n".join(lines)

    return content_processed


def insert_br_html_at_the_end_of_line(lines):
    return [f"{item} <br>" for item in lines]


def get_monitor_desc(param_grid_model_desc: dict):
    result = ""

    for key, value in param_grid_model_desc.items():
        key_processed = process_param_name(key)
        line = f"{key_processed}: {value}<br>"
        result += line

    return result


def process_param_name(name):
    if len(name) == 3:
        return name

    if len(name) > 3:
        return name[:3]

    return name + "_" * (3 - len(name))


# file_path = "artifacts/monitor_desc/monitor_desc.txt"
# print(get_monitor_desc(file_path))


def get_param_grid_model(param_grid_model: dict):
    values = param_grid_model.values()

    values = [myfuncs.get_range_for_param(str(item)) for item in values]

    return dict(zip(list(param_grid_model.keys()), values))


models = {
    "LR": LogisticRegression(random_state=42),
    "LRe": LogisticRegression(penalty="elasticnet", solver="saga", random_state=42),
    "LR1": LogisticRegression(penalty="l1", solver="saga", random_state=42),
    "LSVC": LinearSVC(random_state=42),
    "SVCP": SVC(kernel="poly", degree=2, random_state=42),
    "SVCR": SVC(kernel="rbf", random_state=42),
    "RF": RandomForestClassifier(random_state=42),
    "ET": ExtraTreesClassifier(random_state=42),
    "GB": GradientBoostingClassifier(random_state=42),
    "SGD": GradientBoostingClassifier(random_state=42),
    "XGB": XGBClassifier(random_state=42),
    "LGB": LGBMClassifier(verbose=-1, random_state=42),
}


def get_base_model(model_name: str):
    """Get the Model object from model_name <br>


    Args:
        model_name (str): format = model_name_real_blabla

    """

    model_name_real = model_name.split("_")[0]

    return models[model_name_real]


def sub_param_for_yaml_file(src_path: str, des_path: str, replace_dict: dict):
    """Substitue params in src_path and save in des_path

    Args:
        replace_dict (dict): key: item needed to replace, value: item to replace
        VD:
        ```python
        replace_dict = {
            "${P}": data_transformation,
            "${T}": model_name,
            "${E}": evaluation,
        }

        ```
    """

    with open(src_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    config_str = yaml.dump(config_data, default_flow_style=False)

    for key, value in replace_dict.items():
        config_str = config_str.replace(key, value)

    with open(des_path, "w", encoding="utf-8") as file:
        file.write(config_str)

    print(f"Đã thay thế các tham số trong {src_path} lưu vào {des_path}")


def get_real_column_name(column):
    """After using ColumnTransformer, the column name has format = bla__Age, so only take Age"""

    start_index = column.find("__") + 2
    column = column[start_index:]
    return column


def get_real_column_name_from_get_feature_names_out(columns):
    """Take the exact name from the list retrieved by method get_feature_names_out() of ColumnTransformer"""

    return [get_real_column_name(item) for item in columns]
