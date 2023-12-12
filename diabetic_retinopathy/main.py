import logging
from absl import app, flags
from train import Trainer
from evaluation.eval import Evaluation
from input_pipeline import datasets
from input_pipeline.create_TFRecord import create_tfrecords
from utils import utils_params, utils_misc
from models.architectures import *

FLAGS = flags.FLAGS

flags.DEFINE_boolean('train', True, 'Specify whether to train include evaluate or only evaluate with a given run folder.')
# vgg_like mobilenet inceptionV3  inception_resnet_v2  xception CNN_team06 resnet_simple
flags.DEFINE_string('model_name', 'inceptionV3', 'Name of the models')
flags.DEFINE_string('eval_folder_name', 'run1_2022-02-12T16-03-41-724633',
                    'folder name that you want evaluate, only will be used with flags --train is False')

def main(argv):
    # set folder and gin
    if FLAGS.train:
        run_paths = utils_params.gen_run_folder()
        gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    else:
        run_paths = utils_params.gen_run_folder(FLAGS.eval_folder_name)
        gin.parse_config_files_and_bindings([run_paths['path_eval_gin']], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO, b_stream=False)

    # setup input pipeline
    if create_tfrecords():
        logging.info(
            f"Created TFRecords files at: {gin.query_parameter('create_tfrecords.record_dir')}")
    else:
        logging.info(
            "TFRecords files already exist. Proceed with the execution")

    ds_train, ds_val, ds_test, ds_info = datasets.load(
        data_dir=gin.query_parameter('create_tfrecords.record_dir'))

    # set a model
    if FLAGS.model_name == "vgg_like":
        model = vgg_like(input_shape=(256, 256, 3), n_classes=2)
    elif FLAGS.model_name == "CNN_team06":
        model = CNN_team06()
    elif FLAGS.model_name == "resnet_simple":
        model = resnet_simple()
    else:
        model = Transfer_module(num_classes=2, dropout_rate=0.2, module_name=FLAGS.model_name)

    model.summary()

    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_test, ds_info, run_paths)
        for _ in trainer.train():
            continue

        evaluator = Evaluation(model, ds_test, run_paths)
        evaluator.evaluate()
    else:
        # only evaluate the given checkpoint and given config_operative.gin
        evaluator = Evaluation(model, ds_test, run_paths)
        evaluator.evaluate()


if __name__ == "__main__":
    app.run(main)
