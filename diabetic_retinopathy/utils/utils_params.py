import os
import datetime
import logging


def gen_run_folder(eval_folder_name=''):
    # return a dictionary
    run_paths = dict()
    if not os.path.isdir(eval_folder_name):
        path_model_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'experiments'))
        # creating time
        date_creation = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')

        if not eval_folder_name:
            run_id = 'run1_' + date_creation
            run_paths['path_model_id'] = os.path.join(path_model_root, run_id)
            run_paths['path_ckpts_eval'] = os.path.join(run_paths['path_model_id'], 'ckpts', 'eval')
            run_paths['path_gin'] = os.path.join(run_paths['path_model_id'], 'config_operative.gin')
            run_paths['path_logs_train'] = os.path.join(run_paths['path_model_id'], 'logs', 'run.log')
            run_paths['path_ckpts_train'] = os.path.join(run_paths['path_model_id'], 'ckpts')
        else:
            # given the ecalusation folder name
            run_id = 'run1_eval_' + date_creation
            run_paths['path_model_id'] = os.path.join(path_model_root, run_id)
            run_paths['path_ckpts_eval'] = os.path.join(run_paths['path_model_id'], 'ckpts', 'eval')
            run_paths['path_gin'] = os.path.join(run_paths['path_model_id'], 'config_operative.gin')
            run_paths['given_folder'] = os.path.join(path_model_root, eval_folder_name)
            run_paths['path_logs_train'] = os.path.join(run_paths['path_model_id'], 'logs', 'run_eval.log')
            run_paths['path_ckpts_train'] = os.path.join(run_paths['given_folder'], 'ckpts')
            run_paths['path_eval_gin'] = os.path.join(run_paths['given_folder'], 'config_operative.gin')

    else:
        logging.info("please set flags --eval_folder_name an existed folder name, not a path")
        raise ValueError

    # Create folders
    for k, v in run_paths.items():

        if any([x in k for x in ['path_model', 'path_ckpts']]):
            if not os.path.exists(v):
                os.makedirs(v, exist_ok=True)

    # Create files
    for k, v in run_paths.items():
        if any([x in k for x in ['path_logs']]):
            if not os.path.exists(v):
                os.makedirs(os.path.dirname(v), exist_ok=True)
                # ‘a’ open for writing, appending to the end of the file if it exists
                # ‘w’ open for writing, truncating the file first
                with open(v, 'a'):
                    pass  # atm file creation is sufficient
    return run_paths


"""
saving gin configs in files
"""

def save_config(path_gin, config):
    with open(path_gin, 'w') as f_config:
        f_config.write(config)


def gin_config_to_readable_dictionary(gin_config: dict):
    """
    Parses the gin configuration to a dictionary. Useful for logging to e.g. W&B
    :param gin_config: the gin's config dictionary. Can be obtained by gin.config._OPERATIVE_CONFIG
    :return: the parsed (mainly: cleaned) dictionary
    """
    data = {}
    for key in gin_config.keys():
        name = key[1].split(".")[1]
        values = gin_config[key]
        for k, v in values.items():
            data["/".join([name, k])] = v

    return data
