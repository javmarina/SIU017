import sys

from object_detection import inputs
from object_detection.builders import model_builder

from model_builder import *


class ModelEvaluator(Base):
    def evaluate(self):
        print("Running evaluation loop...")
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model_lib_v2.eval_continuously(
                pipeline_config_path=os.path.join(self._training_loop_path, "pipeline.config"),
                model_dir=self._training_loop_path,
                checkpoint_dir=self._training_loop_path
            )

    def compare_checkpoints(self):
        pass

        # TODO
        configs = config_util.get_configs_from_pipeline_file(self._pipeline_config_path)
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        # latest_checkpoint = tf.train.load_checkpoint(ckpt_dir_or_file=self._training_loop_path)

        # ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)

        # detection_model.summary()

        eval_input_config = configs['eval_input_configs'][0]

        strategy = tf.compat.v2.distribute.get_strategy()
        eval_input = strategy.experimental_distribute_dataset(
            inputs.eval_input(
                eval_config=configs['eval_config'],
                eval_input_config=eval_input_config,
                model_config=configs['model'],
                model=detection_model))

        global_step = tf.compat.v2.Variable(
            0, trainable=False, dtype=tf.compat.v2.dtypes.int64)

        for ckpt_path in [tf.train.get_checkpoint_state(self._training_loop_path).all_model_checkpoint_paths[0]]:
            ckpt = tf.compat.v2.train.Checkpoint(step=global_step, model=detection_model)

            ckpt.restore(ckpt_path).expect_partial()

            summary_writer = tf.compat.v2.summary.create_file_writer(
                os.path.join(self._training_loop_path, 'eval', eval_input_config.name))
            with summary_writer.as_default():
                model_lib_v2.eager_eval_loop(
                    detection_model,
                    configs,
                    eval_input,
                    global_step=global_step)

        # manager = tf.compat.v2.train.CheckpointManager(ckpt, self._training_loop_path, max_to_keep=None)


if __name__ == "__main__":
    model_path = "my_model3"  # sys.argv[1]
    model_evaluator = ModelEvaluator(model_path)
    model_evaluator.evaluate()
    # model_evaluator.compare_checkpoints()
