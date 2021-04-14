# TWINBOT-Python

Personal project for the SIU017 and SIU039 courses (Master's Degree in Intelligent Systems, Jaume I University).

## Branches

See [here](https://github.com/javmarina/SIU017/branches).

* ``master``: original source code for the original Unity simulator.
* ``simulator2``: ``master`` branch adapted to new simulator (v2).
* ``leader-follower``: vision-based leader-follower system.
* ``collaborative-grasping``: grasping using object axes of inertia.

## Installation

    pip install -r requirements.txt

If you also want to use the code inside ``neural_network`` folder, install TensorFlow 2, imgaug and the
[Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)

## How to run

Select the branch with ``git checkout`` and run ``python main.py``
