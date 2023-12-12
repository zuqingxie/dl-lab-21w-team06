import gin
import logging
from absl import app, flags
from ensemble import Ensemble
from train import Trainer
from evaluation.eval import Evaluation
from utils import utils_params, utils_misc
from input_pipeline import datasets
from models.lstm import *
from input_pipeline.creat_TFRecords import create_tfrecords

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train include evaluate or only evaluate with a given run '
                                     'folder.')
flags.DEFINE_string('device_name', 'local', 'setting different path for GPU-Server or local')

# model:'rnn_mix', 'rnn'
flags.DEFINE_string('model_name', 'rnn', 'Name of the models')

flags.DEFINE_string('eval_folder_name', 'run2_2022-02-07T16-03-20-452901',
                    'folder name that you want evaluate, only will be used with flags --train is False')

flags.DEFINE_integer('index_ckpt', 2,
                     'index of the checkpoint that needed to be evaluate, default "0" means latest_checkpoint')

# ensemble_learning
flags.DEFINE_boolean('ensemble_learning', True, 'whether we do an emsemble_learning or not')


@gin.configurable
def main(argv):
    # Ensemble learning or not
    if FLAGS.ensemble_learning:
        run_paths = utils_params.gen_run_folder()
        # gin config definition
        gin.parse_config_files_and_bindings(['configs/config_ensemble.gin'], [])
        if create_tfrecords():
            logging.info(
                f"Created TFRecords files at: {gin.query_parameter('create_tfrecords.records_dir')}")
        else:
            logging.info(
                "TFRecords files already exist. Proceed with the execution")
        
        # set logger
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
        ensemble = Ensemble(run_paths=run_paths, num_classes=12)
        ensemble.average_evaluate()
    else:
        if FLAGS.train:
            run_paths = utils_params.gen_run_folder()
            gin.parse_config_files_and_bindings(['configs/config.gin'], [])
        else:
            run_paths = utils_params.gen_run_folder(FLAGS.eval_folder_name)
            gin.parse_config_files_and_bindings([run_paths['path_eval_gin']], [])
        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        # setup pipeline
        if create_tfrecords():
            logging.info(
                f"Created TFRecords files at: {gin.query_parameter('create_tfrecords.records_dir')}")
        else:
            logging.info(
                "TFRecords files already exist. Proceed with the execution")

        # load the dataset
        ds_train, ds_val, ds_test = datasets.load(name="hapt",
                                                  data_dir=gin.query_parameter('create_tfrecords.records_dir'),
                                                  window_length=gin.query_parameter('create_tfrecords.window_length'),
                                                  window_shift=gin.query_parameter('create_tfrecords.window_shift'))

        # set a model
        if FLAGS.model_name == "rnn_mix":
            model = rnn_mix()
        elif FLAGS.model_name == "rnn":
            model = rnn(window_size=gin.query_parameter('create_tfrecords.window_length'))
        else:
            raise ValueError
        model.summary()

        # train or evaluate
        if FLAGS.train:

            trainer = Trainer(model, ds_train, ds_val, run_paths)
            for _ in trainer.train():
                continue

        evaluator = Evaluation(model, ds_test, run_paths, num_classes=12)
        evaluator.evaluate()


if __name__ == "__main__":
    app.run(main)
