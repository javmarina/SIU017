import glob
import os
import shutil
import tarfile
import urllib.request

DOWNLOAD_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz"
MODEL_NAME = "efficientdet_d1_coco17_tpu-32"
# "http://download.tensorflow.org/models/object_detection/"

if __name__ == "__main__":
    MODEL_FILE = MODEL_NAME + ".tar.gz"
    DEST_DIR = MODEL_NAME

    if not os.path.exists(MODEL_FILE):
        urllib.request.urlretrieve(DOWNLOAD_URL, MODEL_FILE)

    if os.path.exists(MODEL_NAME):
        shutil.rmtree(MODEL_NAME)

    tar = tarfile.open(MODEL_FILE)
    tar.extractall()
    tar.close()

    os.remove(MODEL_FILE)

    fine_tune_checkpoint = os.path.join(DEST_DIR, "model.ckpt")

    exported_path = os.path.join(MODEL_NAME, "exported")
    os.mkdir(exported_path)
    shutil.move(os.path.join(MODEL_NAME, "saved_model"), os.path.join(exported_path, "saved_model"))
    shutil.copytree(os.path.join(MODEL_NAME, "checkpoint"), os.path.join(exported_path, "checkpoint"))
    shutil.copyfile(os.path.join(MODEL_NAME, "pipeline.config"), os.path.join(exported_path, "pipeline.config"))

    training_loop_path = os.path.join(MODEL_NAME, "training_loop")
    os.mkdir(training_loop_path)
    shutil.move(os.path.join(MODEL_NAME, "pipeline.config"), os.path.join(training_loop_path, "pipeline.config"))
    files = glob.glob(os.path.join(MODEL_NAME, "checkpoint", "*"))
    for file in files:
        shutil.move(file, training_loop_path)
    os.rmdir(os.path.join(MODEL_NAME, "checkpoint"))
