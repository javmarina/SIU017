import os
import xml.etree.ElementTree as ET


class Object:
    def __init__(self, name: str, xmin: int, ymin: int, xmax: int, ymax: int,
                 pose: str = "Unspecified", truncated=None, difficult: bool = False,
                 occluded: bool = False, min_dimension=None):
        self.name = name

        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin

        if min_dimension is not None:
            if xmax-xmin < min_dimension:
                diff = min_dimension-(xmax-xmin)
                xmax += int(diff / 2)
                xmin -= int(diff / 2)
            if ymax-ymin < min_dimension:
                diff = min_dimension-(ymax-ymin)
                ymax += int(diff / 2)
                ymin -= int(diff / 2)

        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.pose = pose
        self.truncated = truncated
        self.difficult = difficult
        self.occluded = occluded


class LabeledImage:
    def __init__(self, path: str, shape, database: str = "Unknown", segmented: bool = False):
        self.path = path
        self.width = shape[1]
        self.height = shape[0]
        self.depth = shape[2]
        self.database = database
        self.segmented = segmented
        self.objects = []

    def add_object(self, object: Object):
        if object.xmin <= 0:
            object.xmin = 0
            object.truncated = True
        if object.ymin <= 0:
            object.ymin = 0
            object.truncated = True
        if object.xmax >= self.width:
            object.xmax = self.width
            object.truncated = True
        if object.ymax >= self.height:
            object.ymax = self.height
            object.truncated = True

        if object.truncated is None:
            object.truncated = False
        self.objects.append(object)


class Reader:
    def __init__(self, path: str):
        self._path = path
        self._labeled_img = None

    def get_labeled_img(self) -> LabeledImage:
        if self._labeled_img is None:
            root = ET.parse(self._path).getroot()
            self._labeled_img = self._extract_img(root)
        return self._labeled_img

    @staticmethod
    def _extract_img(root: ET.Element) -> LabeledImage:
        path = root.find("model_path").text

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        depth = int(size.find("depth").text)
        shape = (height, width, depth)

        database = root.find("source/database").text
        segmented = bool(int(root.find("segmented").text))

        labeled_img = LabeledImage(path, shape, database, segmented)

        for object_child in root.findall("object"):
            name = object_child.find("name").text

            difficult_child = object_child.find("difficult")
            difficult = bool(int(difficult_child.text)) if difficult_child is not None else False

            truncated_child = object_child.find("truncated")
            truncated = bool(int(truncated_child.text)) if truncated_child is not None else False

            occluded_child = object_child.find("occluded")
            occluded = bool(int(occluded_child.text)) if occluded_child is not None else False

            pose = object_child.find("pose").text

            bndbox = object_child.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            labeled_img.add_object(Object(name=name, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                                          pose=pose, truncated=truncated, difficult=difficult, occluded=occluded))

        return labeled_img


class Writer:
    def __init__(self, labeled_img: LabeledImage):
        abspath = os.path.abspath(labeled_img.path)

        top = ET.Element("annotation")

        ET.SubElement(top, "folder").text = os.path.basename(os.path.dirname(abspath))
        ET.SubElement(top, "record_filename").text = os.path.basename(abspath)
        ET.SubElement(top, "model_path").text = abspath

        source_child = ET.SubElement(top, "source")
        ET.SubElement(source_child, "database").text = labeled_img.database

        size_child = ET.SubElement(top, "size")
        ET.SubElement(size_child, "width").text = str(labeled_img.width)
        ET.SubElement(size_child, "height").text = str(labeled_img.height)
        ET.SubElement(size_child, "depth").text = str(labeled_img.depth)

        ET.SubElement(top, "segmented").text = str(labeled_img.segmented & 1)

        for object in labeled_img.objects:
            object_child = ET.SubElement(top, "object")

            ET.SubElement(object_child, "name").text = object.name
            ET.SubElement(object_child, "pose").text = object.pose
            ET.SubElement(object_child, "truncated").text = str(object.truncated & 1)
            ET.SubElement(object_child, "difficult").text = str(object.difficult & 1)
            ET.SubElement(object_child, "occluded").text = str(object.occluded & 1)

            bndbox_child = ET.SubElement(object_child, "bndbox")
            ET.SubElement(bndbox_child, "xmin").text = str(object.xmin)
            ET.SubElement(bndbox_child, "ymin").text = str(object.ymin)
            ET.SubElement(bndbox_child, "xmax").text = str(object.xmax)
            ET.SubElement(bndbox_child, "ymax").text = str(object.ymax)

        self._top = top

    def save(self, annotation_path: str):
        with open(annotation_path, mode="w", encoding="utf-8") as f:
            f.write(ET.tostring(self._top, encoding="unicode", xml_declaration=False))
