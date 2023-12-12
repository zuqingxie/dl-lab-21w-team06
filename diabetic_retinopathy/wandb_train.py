import logging
from train import Trainer
from evaluation.eval import Evaluation
from input_pipeline import datasets
from input_pipeline.create_TFRecord import create_tfrecords
from utils import utils_params, utils_misc
from models.architectures import *
import wandb

def train_func():
    with wandb.init() as run:
        gin.clear_config()
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')

        # generate folder structures
        logging.info("----------------------------let us begin--------------------------")
        run_paths = utils_params.gen_run_folder()
        gin.parse_config_files_and_bindings(['configs/config.gin'], [])

        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO, b_stream=False)

        # setup wandb
        wandb.login(key='74eb390cd2dd2a04bf50cef29985133c354877bd')
        wandb.init(project="diabetic_retinopathy", entity="zcz", name=run_paths["path_model_id"],
                   config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

        # setup pipeline
        if create_tfrecords():
            logging.info(
                f"Created TFRecords files at: {gin.query_parameter('create_tfrecords.record_dir')}")
        else:
            logging.info(
                "TFRecords files already exist. Proceed with the execution")

        ds_train, ds_val, ds_test, ds_info = datasets.load(
            data_dir=gin.query_parameter('create_tfrecords.record_dir'))

        # setup model
        model = vgg_like(input_shape=(256, 256, 3), n_classes=2)
        model.summary()

        # start train and sweep
        logging.info("---------------------------training--------------------------")
        trainer = Trainer(model, ds_train, ds_val, ds_test, ds_info, run_paths)
        for _ in trainer.train():
            continue

        evaluator = Evaluation(model, ds_test, run_paths)
        evaluator.evaluate()

train_func()