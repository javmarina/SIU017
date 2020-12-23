import glob
import math
import os
import random
import subprocess
import sys
import time
import webbrowser
from itertools import zip_longest
from shutil import copyfile, rmtree

import cv2 as cv
import imgaug
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from google.protobuf import text_format
from imgaug import augmenters as iaa
from object_detection import model_lib_v2, exporter_lib_v2
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMapItem, StringIntLabelMap
from object_detection.utils import dataset_util, label_map_util, config_util

# In order for below imports to work
sys.path.append("..")

from ImagePipeline import ImagePipeline
from RobotHttpInterface import RobotHttpInterface
from RobotModel import RobotModel
from neural_network.PascalVoc import Object, LabeledImage, Reader, Writer


class Base:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise ValueError("{:s} folder not found".format(model_path))

        self._imgs_path = os.path.join(model_path, "imgs")
        if not os.path.exists(self._imgs_path):
            os.makedirs(self._imgs_path)

        self._xml_path = os.path.join(model_path, "xml")
        if not os.path.exists(self._xml_path):
            os.makedirs(self._xml_path)

        self._imgs_aug_path = os.path.join(model_path, "imgs_aug")
        if not os.path.exists(self._imgs_aug_path):
            os.makedirs(self._imgs_aug_path)

        self._xml_aug_path = os.path.join(model_path, "xml_aug")
        if not os.path.exists(self._xml_aug_path):
            os.makedirs(self._xml_aug_path)

        self._annotations_path = os.path.join(model_path, "annotations")
        if not os.path.exists(self._annotations_path):
            os.makedirs(self._annotations_path)

        self._label_map_path = os.path.join(self._annotations_path, "label_map.pbtxt")
        if not os.path.exists(self._label_map_path):
            self._label_map_dict = {}
            self._save_label_dict_to_file(self._label_map_dict, self._label_map_path)
        else:
            self._label_map_dict = label_map_util.get_label_map_dict(self._label_map_path)

        self._train_path = os.path.join(model_path, "train")
        self._test_path = os.path.join(model_path, "test")

        self._exported_path = os.path.join(model_path, "exported")
        if not os.path.exists(self._exported_path):
            os.makedirs(self._exported_path)

        self._training_loop_path = os.path.join(model_path, "training_loop")
        if not os.path.exists(self._training_loop_path):
            os.makedirs(self._training_loop_path)

        self._pipeline_config_path = os.path.join(self._training_loop_path, "pipeline.config")
        if not os.path.exists(self._pipeline_config_path):
            raise FileNotFoundError("pipeline.config file must be present in {:s}"
                                    .format(self._pipeline_config_path))

        self._img_filenames = glob.glob(os.path.join(self._imgs_path, "*.jpg"))
        self._xml_filenames = glob.glob(os.path.join(self._xml_path, "*.xml"))

        # Verification
        img_count = len(self._img_filenames)
        xml_count = len(self._xml_filenames)
        if img_count != xml_count:
            raise RuntimeError("There are {:d} images but {:d} XML files.".format(img_count, xml_count))

        self._indexes = []
        for img_path, xml_path in zip(self._img_filenames, self._xml_filenames):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            xml_name = os.path.splitext(os.path.basename(xml_path))[0]
            if img_name != xml_name:
                raise RuntimeError("")
            else:
                self._indexes.append(int(img_name))
        self._src_file_count = img_count
        self._next_file_index = max(self._indexes, default=-1) + 1

    @staticmethod
    def _save_label_dict_to_file(label_dict: dict, label_map_path: str):
        label_map = StringIntLabelMap()
        for label, id in label_dict.items():
            label_map.item.append(StringIntLabelMapItem(id=id, name=label))
        with open(label_map_path, "w") as f:
            f.write(str(text_format.MessageToBytes(label_map, as_utf8=True), "utf-8"))

    def add_annotation(self, class_label: str):
        if class_label not in self._label_map_dict:
            max_id = max([id for label, id in self._label_map_dict.items()], default=0)
            self._label_map_dict[class_label] = max_id + 1
            # Update file
            self._save_label_dict_to_file(self._label_map_dict, self._label_map_path)

    def _get_next_file_index(self) -> int:
        return self._next_file_index

    def _get_src_file_count(self) -> int:
        return self._src_file_count

    def _get_aug_file_count(self) -> int:
        return len(glob.glob(os.path.join(self._imgs_aug_path, "*.jpg")))

    def _iterate_source_files(self):
        return zip(self._indexes, self._img_filenames, self._xml_filenames)

    def _iterate_aug_files(self):
        return zip(glob.glob(os.path.join(self._imgs_aug_path, "*.jpg")),
                   glob.glob(os.path.join(self._xml_aug_path, "*.xml")))

    def _img_filename_for_index(self, file_index: int) -> str:
        return os.path.join(self._imgs_path, "{:d}.jpg".format(file_index))

    def _xml_filename_for_index(self, file_index: int) -> str:
        return os.path.join(self._xml_path, "{:d}.xml".format(file_index))

    def _remove_file_index(self, file_index: int):
        os.remove(self._img_filename_for_index(file_index))
        os.remove(self._xml_filename_for_index(file_index))


class SampleAdquisition(Base):
    def __init__(self, robot_model: RobotModel, model_path: str, reset=False, address="127.0.0.1"):
        super().__init__(model_path)
        self._adq_rate = 10
        self._controller = RobotHttpInterface(robot_model, address)
        self._run = True
        if reset:
            self._current_index = 0
        else:
            self._current_index = self._get_next_file_index()
            if self._current_index > 0:
                # Images already acquired, skip step
                self._run = False
        self._pipeline = ImagePipeline(address, robot_model, self._adq_rate)

    def run(self):
        if not self._run:
            print("Images already acquired, skipping SampleAdquisition step")
            return

        self._pipeline.start()
        time.sleep(1)
        while True:
            time.sleep(1 / self._adq_rate)
            output = self._pipeline[-2].get_last_output()
            if output is None:
                continue
            img, tube = output
            self._process_frame(img, tube)

            if self._pipeline[-1].is_stopped():
                break
        print("Finalizado!")
        self._controller.stop()
        self._pipeline.stop()

    def _process_frame(self, img: np.array, tube):
        if tube is None:
            return
        x, y, w, h = cv.boundingRect(tube)

        img_path = self._img_filename_for_index(self._current_index)
        xml_path = self._xml_filename_for_index(self._current_index)

        # TODO: extend bounding box some pixels?

        labeled_img = LabeledImage(img_path, img.shape)
        labeled_img.add_object(Object("tube", x, y, x + w, y + h, min_dimension=30))
        self.add_annotation("tube")

        writer = Writer(labeled_img)
        writer.save(xml_path)

        # Save image
        Image.fromarray(img).save(img_path)

        print("Saved file", self._current_index)
        self._current_index += 1


class SampleVerification(Base):
    def __init__(self, model_path):
        super().__init__(model_path)
        self._current_indexes = []
        self._current_axes = []

    def run(self):
        if len(glob.glob(os.path.join(self._imgs_aug_path, "*.jpg"))) > 0:
            # Images already augmented (therefore verified), skip step
            print("Images already verified, skipping SampleVerification step")
            return

        count = 0
        for chunk in self.grouper(self._iterate_source_files(), 12, fillvalue=None):
            objects = []
            imgs = []
            self._current_indexes = []
            subtitles = []
            for item in chunk:
                if item is None:
                    continue
                i, img_filename, xml_filename = item
                self._current_indexes.append(i)
                reader = Reader(xml_filename)
                labeled_img = reader.get_labeled_img()
                objects.append(labeled_img.objects[0])
                imgs.append(np.array(Image.open(img_filename)))
                subtitles.append(img_filename)

            progress = 100 * count / self._get_src_file_count()
            self._current_axes = self._show_grid(imgs, objects, 3, 4, title="{:.1f}% completado".format(progress),
                                                 subtitles=subtitles)

            mng = plt.get_current_fig_manager()
            mng.window.state("zoomed")
            plt.show(block=True)
            count += len(objects)
        self._current_indexes = []
        self._current_axes = []

    @staticmethod
    def grouper(iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return zip_longest(*args, fillvalue=fillvalue)

    @staticmethod
    def _press(event):
        if event.key == "enter":
            plt.close()

    def _onclick(self, event):
        for i, ax in enumerate(self._current_axes):
            if ax == event.inaxes:
                file_index = self._current_indexes[i]
                print("Removed file", file_index)
                self._remove_file_index(file_index)
                ax.clear()
                plt.draw()
                break

    def _show_grid(self, imgs, objects, m, n, title="", subtitles=None):
        """
        Mostrar las imágenes en subplots de la misma figura.
        :param imgs: imágenes a mostrar
        :param m: preferencia de número de filas
        :param n: preferencia de número de columnas
        :param title: título global de la figura
        :param subtitles: subtítulo de cada subfigura (una por imagen)
        :return: lista de axes
        """
        N = len(imgs)

        # print(m,n)
        fig = plt.figure(figsize=(m, n))
        fig.canvas.mpl_connect("button_press_event", self._onclick)
        fig.canvas.mpl_connect("key_press_event", self._press)
        plt.gray()
        axes = []
        for i in range(N):
            ax = fig.add_subplot(m, n, i + 1)
            axes.append(ax)
            ax.imshow(imgs[i])
            object = objects[i]
            ax.add_patch(
                patches.Rectangle(
                    (object.xmin, object.ymin),
                    object.xmax - object.xmin,
                    object.ymax - object.ymin,
                    edgecolor="blue",
                    facecolor="none"
                ))
            if subtitles is not None:
                ax.set_title(subtitles[i])

        fig.suptitle(title)
        return axes


class DataAugmenter(Base):
    def __init__(self, model_path):
        super().__init__(model_path)
        # Add iaa.Rotate(0) to list if you want to keep the original image
        self._augmenters = [iaa.Rotate((-180, 180), fit_output=False, mode="edge")] * 5

    def augment_data(self):
        if len(glob.glob(os.path.join(self._imgs_aug_path, "*.jpg"))) > 0:
            # Images already augmented, skip step
            print("Images already augmented, skipping DataAugmenter step")
            return

        augmenters = self._augmenters
        for i, img_path, xml_path in self._iterate_source_files():
            im_np = np.array(Image.open(img_path))
            objects = Reader(xml_path).get_labeled_img().objects
            bounding_boxes = [imgaug.BoundingBox(x1=object.xmin, y1=object.ymin,
                                                 x2=object.xmax, y2=object.ymax, label=object.name)
                              for object in objects]
            bbs = imgaug.BoundingBoxesOnImage(bounding_boxes, shape=im_np.shape)
            for j in range(len(augmenters)):
                image_aug, bbs_aug = augmenters[j].augment(image=im_np, bounding_boxes=bbs)
                new_img_path = os.path.join(self._imgs_aug_path, "{}_{}.jpg".format(i, j))
                new_xml_path = os.path.join(self._xml_aug_path, "{}_{}.xml".format(i, j))
                labeled_img = LabeledImage(path=new_img_path,
                                           shape=im_np.shape)
                for bb_aug in bbs_aug.bounding_boxes:
                    labeled_img.add_object(Object(name=bb_aug.label,
                                                  xmin=int(bb_aug.x1),
                                                  ymin=int(bb_aug.y1),
                                                  xmax=int(bb_aug.x2),
                                                  ymax=int(bb_aug.y2)))

                Writer(labeled_img).save(new_xml_path)
                Image.fromarray(image_aug).save(new_img_path)
                print("Saved {} and {}".format(new_img_path, new_xml_path))


class TfRecordGenerator(Base):
    def generate_tf_records(self, ratio: int = 0.1):
        skipped = self._copy_files(ratio)
        if not skipped:
            self._generate_tf_record(self._test_path, "test.record")
            self._generate_tf_record(self._train_path, "train.record")

    def _copy_files(self, ratio) -> bool:
        if os.path.exists(os.path.join(self._train_path, "train.record")):
            # Train TF record already generated (should also have test TF record), skip step
            print("TF records already generated, skipping TfRecordGenerator step")
            return True

        if os.path.exists(self._train_path):
            rmtree(self._train_path)
            print("Removed {} folder".format(self._train_path))
        os.makedirs(self._train_path)

        if os.path.exists(self._test_path):
            rmtree(self._test_path)
            print("Removed {} folder".format(self._test_path))
        os.makedirs(self._test_path)

        aug_file_info = list(self._iterate_aug_files())
        num_images = len(aug_file_info)
        num_test_images = math.ceil(ratio * num_images)

        # Copy randomly selected files to test/ folder
        for i in range(num_test_images):
            idx = random.randint(0, num_images - 1)
            img_path, xml_path = aug_file_info[idx]

            img_filename = os.path.basename(img_path)
            new_img_path = os.path.join(self._test_path, img_filename)
            copyfile(img_path, new_img_path)

            xml_filename = os.path.basename(xml_path)
            new_xml_path = os.path.join(self._test_path, xml_filename)
            copyfile(xml_path, new_xml_path)

            del aug_file_info[idx]
            num_images -= 1
        print("Files copied to {}".format(self._test_path))

        # Copy remaining files to train/ folder
        for img_path, xml_path in aug_file_info:
            img_filename = os.path.basename(img_path)
            new_img_path = os.path.join(self._train_path, img_filename)
            copyfile(img_path, new_img_path)

            xml_filename = os.path.basename(xml_path)
            new_xml_path = os.path.join(self._train_path, xml_filename)
            copyfile(xml_path, new_xml_path)
        print("Files copied to {}".format(self._train_path))
        return False

    def _generate_tf_record(self, path: str, record_filename: str):
        # TODO: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#sharding-datasets
        xml_files = glob.glob(os.path.join(path, "*.xml"))
        labeled_imgs = [Reader(xml_file).get_labeled_img() for xml_file in xml_files]

        record_path = os.path.join(path, record_filename)
        writer = tf.io.TFRecordWriter(record_path)
        for labeled_img in labeled_imgs:
            tf_example = self.create_tf_example(labeled_img)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print("Successfully created the TFRecord file: {}".format(record_path))

    def create_tf_example(self, labeled_img: LabeledImage):
        height = labeled_img.height  # Image height
        width = labeled_img.width  # Image width
        path = labeled_img.path
        filename = os.path.basename(path).encode("utf8")  # Filename of the image. Empty if image is not from file
        image_format = os.path.splitext(filename)[1][1:]  # b"jpeg" or b"png"

        with tf.io.gfile.GFile(path, "rb") as fid:
            encoded_jpg = fid.read()  # Encoded image bytes

        xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
        xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
        ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
        ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
        classes_text = []  # List of string class name of bounding box (1 per box)
        classes = []  # List of integer class id of bounding box (1 per box)
        for object in labeled_img.objects:
            xmins.append(object.xmin / width)
            xmaxs.append(object.xmax / width)
            ymins.append(object.ymin / height)
            ymaxs.append(object.ymax / height)
            classes_text.append(object.name.encode("utf8"))
            classes.append(self._label_map_dict[object.name])

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            "image/height": dataset_util.int64_feature(height),
            "image/width": dataset_util.int64_feature(width),
            "image/record_filename": dataset_util.bytes_feature(filename),
            "image/source_id": dataset_util.bytes_feature(filename),
            "image/encoded": dataset_util.bytes_feature(encoded_jpg),
            "image/format": dataset_util.bytes_feature(image_format),
            "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
            "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
            "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
            "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
            "image/object/class/text": dataset_util.bytes_list_feature(classes_text),
            "image/object/class/label": dataset_util.int64_list_feature(classes),
        }))
        return tf_example


class ModelTrainer(Base):
    def __init__(self, model_path: str, num_steps: int, memory_growth: bool = False):
        super().__init__(model_path)
        self._num_steps = num_steps
        self._memory_growth = memory_growth

    def run(self):
        self._validate_pipeline_config()

        if self._memory_growth:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e)

        print("Running train loop...")
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model_lib_v2.train_loop(
                pipeline_config_path=self._pipeline_config_path,
                model_dir=self._training_loop_path
            )

    def _validate_pipeline_config(self):
        configs = config_util.get_configs_from_pipeline_file(self._pipeline_config_path)

        model_config = configs["model"]
        self.assert_equals(
            model_config.ssd.num_classes,
            len(self._label_map_dict)
        )

        train_config = configs["train_config"]
        self.assert_equals(
            train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.total_steps,
            self._num_steps
        )
        self.assert_equals(
            train_config.num_steps,
            self._num_steps
        )

        train_input_reader = configs["train_input_config"]
        self.assert_equals(
            train_input_reader.label_map_path,
            self._label_map_path
        )
        self.assert_equals(
            train_input_reader.tf_record_input_reader.input_path[0],
            os.path.join(self._train_path, "train.record")
        )

        eval_input_reader = configs["eval_input_config"]
        self.assert_equals(
            eval_input_reader.label_map_path,
            self._label_map_path
        )
        self.assert_equals(
            eval_input_reader.tf_record_input_reader.input_path[0],
            os.path.join(self._test_path, "test.record")
        )

    @staticmethod
    def assert_equals(found, expected):
        assert found == expected, "Found '{}', expected '{}'".format(str(found), str(expected))


class ModelExporter(Base):
    def export_model(self):
        configs = config_util.get_configs_from_pipeline_file(self._pipeline_config_path)
        pipeline_config = config_util.create_pipeline_proto_from_configs(configs)

        exporter_lib_v2.export_inference_graph(
            input_type="image_tensor",
            pipeline_config=pipeline_config,
            trained_checkpoint_dir=self._training_loop_path,
            output_directory=self._exported_path
        )


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("TkAgg")

    model_path = input("Write the model path: ")

    adq = SampleAdquisition(RobotModel.GIRONA_500_1, model_path, reset=True)
    adq.run()

    verification = SampleVerification(model_path)
    verification.run()

    data_augmenter = DataAugmenter(model_path)
    data_augmenter.augment_data()

    record_generator = TfRecordGenerator(model_path)
    record_generator.generate_tf_records()

    p1 = subprocess.Popen(["tensorboard", "--logdir={}/".format(model_path)])
    webbrowser.open("http://localhost:6006/")

    # p2 = subprocess.Popen(["python", "model_evaluator.py", model_path])

    model_trainer = ModelTrainer(model_path, num_steps=40000)
    model_trainer.run()

    p1.kill()
    # p2.kill()

    model_exporter = ModelExporter(model_path)
    model_exporter.export_model()
