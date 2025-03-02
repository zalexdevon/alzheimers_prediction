from classifier.Mylib.myfuncs import read_yaml, sub_param_for_yaml_file
from classifier.constants import *
from pathlib import Path
import sys
import os

params = read_yaml(Path(PARAMS_FILE_PATH))
data_transformation = params.data_transformation

model_name = params.model_name
evaluated_data_transformation = params.evaluated_data_transformation
evaluated_model_name = params.evaluated_model_name

replace_dict = {
    "${P}": data_transformation,
    "${T}": model_name,
    "${E}": evaluated_model_name,
    "${PE}": evaluated_data_transformation,
}

sub_param_for_yaml_file("config_p.yaml", "config.yaml", replace_dict)
sub_param_for_yaml_file("dvc_p.yaml", "dvc.yaml", replace_dict)

stage_name = sys.argv[1]
os.system(f"dvc repro {stage_name}")

# Xóa file .ignore nếu có 
file_path = "artifacts/data_transformation/.gitignore"
if os.path.exists(file_path):
    os.remove(file_path)

# Chạy xong trên colab push lên git ngay
os.system("git add .")
os.system('git commit -m "c"')
os.system("git push")
