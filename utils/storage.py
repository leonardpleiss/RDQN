from datetime import datetime
import os
import time

def get_export_path(
        model_name,
        environment_name:str,
        buffer_name:str
        )->str:
    
    time.sleep(1)
    datetime_string = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(environment_name.replace("/", ""))
    parsed_environment_name = environment_name.replace("/", "") # r"/".split(environment_name)[-1] if ("/" in environment_name) else environment_name

    print(os.listdir())
    trial_result_folder = f"{datetime_string}_{model_name}_{parsed_environment_name}_{buffer_name}"
    trial_result_folder_path = f"code_cluster/RDQN/results/{trial_result_folder}/"
    os.makedirs(trial_result_folder_path, exist_ok=True)


    return trial_result_folder_path


def get_tb_storage_file_path(environment_name, replay_buffer_class, model_name):

    storage_file_path = f"./results/tensorboard_logs/{model_name}/{environment_name}/{replay_buffer_class.__name__}/"
    return storage_file_path

