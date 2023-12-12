import gin
import logging
import wandb
from absl import flags
from train import Trainer
from evaluation.eval import Evaluation
from utils import utils_params, utils_misc
from input_pipeline import datasets
from models.lstm import *
from input_pipeline.creat_TFRecords import create_tfrecords

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('device_name', 'local', 'setting different path for GPU-Server or local')

# lstm_simple rnn
flags.DEFINE_string('model_name', 'rnn', 'Name of the models')


@gin.configurable
def train_func():
    with wandb.init() as run:
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup wandb
    wandb.login(key='74eb390cd2dd2a04bf50cef29985133c354877bd')
    wandb.init(project="diabetic_retinopathy", entity="zcz", name=run_paths["path_model_id"],
               config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

    # setup pipeline
    if create_tfrecords():
        logging.info(
            f"Created TFRecords files at: {gin.query_parameter('create_tfrecords.records_dir')}")
    else:
        logging.info(
            "TFRecords files already exist. Proceed with the execution")

    ds_train, ds_val, ds_test = datasets.load(name="hapt",
        data_dir=gin.query_parameter('create_tfrecords.records_dir'),
        window_length = gin.query_parameter('create_tfrecords.window_length'),
        window_shift = gin.query_parameter('create_tfrecords.window_shift'))

    # model
    if FLAGS.model_name == "lstm_simple":
        model = lstm_simple()
    elif FLAGS.model_name == "rnn":
        model = rnn(window_size=gin.query_parameter('create_tfrecords.window_length'))
    else:
        raise ValueError
    model.summary()

    # train or evaluate
    if FLAGS.train:
        logging.info(f'===========================================start '
                     f'training with model: {FLAGS.model_name}. ======================================')
        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue

        evaluator = Evaluation(model, ds_test, run_paths, num_classes=12)
        evaluator.evaluate()
        evaluator.save_total_confusionmatrix()


train_func()