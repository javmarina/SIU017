import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from object_detection import inputs, model_lib_v2
from object_detection.builders import model_builder
from object_detection.utils import config_util

from model_builder import Base


class ModelEvaluator(Base):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self._eval_results_folder = os.path.join(self._training_loop_path, "eval_results")
        if not os.path.exists(self._eval_results_folder):
            os.mkdir(self._eval_results_folder)
        self._eval_results_filename = os.path.join(self._eval_results_folder, "results.p")

        self._detection_model = None
        self._ckpt_paths = []

    def eval_continuously(self):
        print("Running evaluation loop...")
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model_lib_v2.eval_continuously(
                pipeline_config_path=os.path.join(self._training_loop_path, "pipeline.config"),
                model_dir=self._training_loop_path,
                checkpoint_dir=self._training_loop_path
            )

    def evaluate_checkpoints(self):
        print("Loading model... ")

        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(self._pipeline_config_path)
        self._detection_model = model_builder.build(model_config=configs["model"], is_training=False)

        # Load evaluation inputs
        strategy = tf.compat.v2.distribute.get_strategy()
        eval_input = strategy.experimental_distribute_dataset(
            inputs.eval_input(
                eval_config=configs['eval_config'],
                eval_input_config=configs['eval_input_configs'][0],
                model_config=configs['model'],
                model=self._detection_model))

        self._ckpt_paths = tf.train.get_checkpoint_state(self._training_loop_path).all_model_checkpoint_paths

        global_step = tf.compat.v2.Variable(
            0, trainable=False, dtype=tf.compat.v2.dtypes.int64)

        results = []
        if os.path.exists(self._eval_results_filename):
            with open(self._eval_results_filename, "rb") as f:
                results = pickle.load(f)

        calculated_steps = [result[0] for result in results]

        for ckpt_path in self._ckpt_paths:
            # Restore checkpoint
            ckpt = tf.compat.v2.train.Checkpoint(model=self._detection_model, step=global_step)
            ckpt.restore(ckpt_path).expect_partial()

            step_value = int(global_step.read_value())
            if step_value in calculated_steps:
                print("Evaluation for global step {:d} already done, skipping...".format(step_value))
                continue

            print("Running evaluation for checkpoint {}...".format(ckpt_path))
            evaluation = model_lib_v2.eager_eval_loop(
                detection_model=self._detection_model,
                configs=configs,
                eval_dataset=eval_input,
                global_step=global_step
            )
            results.append((step_value, evaluation))
            calculated_steps.append(step_value)

        with open(self._eval_results_filename, "wb") as f:
            pickle.dump(obj=results, file=f)

    def select_best_checkpoints(self, num_checkpoints=7, bPlot=False):
        results = []
        if os.path.exists(self._eval_results_filename):
            with open(self._eval_results_filename, "rb") as f:
                results = pickle.load(f)

        if len(results) == 0:
            return

        keys = results[0][1].keys()
        steps = np.array([result[0] for result in results])

        mAp = np.zeros(shape=(len(steps),))
        mAp_count = 0
        recall = np.zeros(shape=(len(steps),))
        recall_count = 0

        for key in keys:
            values = [result[1][key] for result in results]

            if bPlot:
                plt.plot(steps, values)
                plt.title(key)
                if "Loss" not in key:
                    plt.ylim([0, 1])
                plt.grid()
                plt.xticks(steps)
                plt.show(block=True)

            if "mAP" in key and "small" not in key:
                mAp += values
                mAp_count += 1
            if "Recall" in key:
                recall += values
                recall_count += 1

        mAp /= mAp_count
        recall /= recall_count
        mAp[mAp < 0] = 0
        recall[recall < 0] = 0
        norm = np.sqrt(mAp ** 2 + recall ** 2)

        if bPlot:
            plt.plot(steps, mAp, label="Average mAp")
            plt.plot(steps, recall, label="Average Recall")
            plt.plot(steps, norm, label="Norm")
            plt.legend()
            plt.show(block=True)

        indices = np.argpartition(norm, -num_checkpoints)[-num_checkpoints:]
        max_step_values = steps[indices]

        global_step = tf.compat.v2.Variable(
            0, trainable=False, dtype=tf.compat.v2.dtypes.int64)

        to_remove_paths = []

        for ckpt_path in self._ckpt_paths:
            # Restore checkpoint
            ckpt = tf.compat.v2.train.Checkpoint(model=self._detection_model, step=global_step)
            ckpt.restore(ckpt_path).expect_partial()

            step_value = int(global_step.read_value())
            if step_value not in max_step_values:
                to_remove_paths.append(ckpt_path)
        to_remove_paths = ([path + ".index" for path in to_remove_paths] +
                           [path + ".data-00000-of-00001" for path in to_remove_paths])

        for path in to_remove_paths:
            os.remove(path)


def yes_no_input(msg: str) -> bool:
    response = ""
    while not (response.lower() == "y" or response.lower() == "n"):
        response = input(msg + "? (Y/N) ")
    return response.lower() == "y"


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("TkAgg")

    model_path = input("Write the model path: ")
    use_tensorboard = yes_no_input("Use TensorBoard")

    p1 = None
    if use_tensorboard:
        import subprocess
        import webbrowser

        p1 = subprocess.Popen(["tensorboard", "--logdir={}/".format(model_path)])
        webbrowser.open("http://localhost:6006/")

    model_evaluator = ModelEvaluator(model_path)
    if yes_no_input("Eval continuously"):
        model_evaluator.eval_continuously()
    if yes_no_input("Evaluate existing checkpoints"):
        model_evaluator.evaluate_checkpoints()
    if yes_no_input("Select best checkpoints"):
        num_checkpoints = input("Number of checkpoints to keep: ")
        bPlot = yes_no_input("Show plots")
        model_evaluator.select_best_checkpoints(
            num_checkpoints=int(num_checkpoints),
            bPlot=bPlot
        )

    if use_tensorboard:
        p1.kill()
