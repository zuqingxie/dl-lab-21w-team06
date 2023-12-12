import gin
import tensorflow as tf
import logging
#import wandb

from absl import flags
FLAGS = flags.FLAGS


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, run_paths, total_steps, log_interval, ckpt_interval,learning_rate, beta_2, beta_1):

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.max_acc = 0
        self.acc_this_time = 0
        self.min_loss = 5
        self.min_loss_this_time = 0


        # Loss
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='val_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.checkpoint_path = self.run_paths["path_ckpts_train"]
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=10)

    @tf.function
    def train_step(self, features, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout). inference可以理解成为前向传播
            predictions = self.model(features, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, features, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(features, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def train(self):
        logging.info(f'===========================================start '
                     f'training with model: {FLAGS.model_name}. ======================================')
        for idx, (features, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(features, labels)

            if step % self.log_interval == 0:

                # Reset test metrics
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()

                for test_features, test_labels in self.ds_val:
                    self.test_step(test_features, test_labels)

                template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.test_loss.result(),
                                             self.test_accuracy.result() * 100))

                # wandb logging
                # wandb.log({'train_acc': self.train_accuracy.result() * 100, 'train_loss': self.train_loss.result(),
                #            'val_acc': self.test_accuracy.result() * 100, 'val_loss': self.test_loss.result(),
                #            'step': step})

                # Write summary to tensorboard
                # ...cm here

                # Reset train metrics

                self.acc_this_time = self.test_accuracy.result().numpy()
                self.min_loss_this_time = self.test_loss.result().numpy()

                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.test_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                if self.acc_this_time > self.max_acc and self.min_loss_this_time < self.min_loss:
                    self.max_acc = self.acc_this_time

                    # Save checkpoint
                    self.ckpt_manager.save()
                    logging.info(f'------Saving checkpoint to {self.run_paths["path_ckpts_train"]}.------')

            if step % self.total_steps == 0:
                logging.info(f'===========================================Finished training after {step} steps.======================================')
                # Save final checkpoint

                return self.test_accuracy.result().numpy()
