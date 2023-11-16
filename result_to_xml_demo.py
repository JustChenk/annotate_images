"""
单张图片demo示例
生成的xml文件与本demo在同一目录下

后续扩充为遍历文件夹内所有图片
"""

import os
import cv2
import torch
import torchsnooper
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ultralytics import YOLO


def img_hwd(img_path):
    img = cv2.imread(img_path)
    return img.shape


def sub_text_ele(parent_ele, ele_name, ele_text):
    ele = ET.SubElement(parent_ele, ele_name)
    ele.text = ele_text


def annotate_img_info(annotation, img_path):
    sub_text_ele(annotation, "folder", os.path.dirname(img_path))
    sub_text_ele(annotation, "filename", os.path.basename(img_path))
    sub_text_ele(annotation, "path", img_path)

    source = ET.SubElement(annotation, "source")
    sub_text_ele(source, "database", "Unknown")

    size = ET.SubElement(annotation, "size")
    _height, _width, _depth = img_hwd(img_path)
    sub_text_ele(size, "height", str(_height))
    sub_text_ele(size, "width", str(_width))
    sub_text_ele(size, "depth", str(_depth))

    sub_text_ele(annotation, "segmented", "0")


def annotate_a_class(annotation, cls_tensor):
    if int(int(cls_tensor.item())) == 1:
        sub_text_ele(annotation, "name", "human")
    else:
        sub_text_ele(annotation, "name", "face")


def annotate_a_box(annotation, box_tensor, cls_tensor):
    object_elem = ET.SubElement(annotation, "object")

    # 判断这个box的class
    # sub_text_ele(object_elem, "name", "human")
    if int(int(cls_tensor.item())) == 1:
        sub_text_ele(object_elem, "name", "human")
    else:
        sub_text_ele(object_elem, "name", "face")

    sub_text_ele(object_elem, "pose", "Unspecified")
    sub_text_ele(object_elem, "truncated", "0")
    sub_text_ele(object_elem, "difficult", "0")

    bndbox = ET.SubElement(object_elem, "bndbox")
    for i, _name in enumerate(["xmin", "ymin", "xmax", 'ymax']):
        sub_text_ele(bndbox, _name, str(box_tensor[i].numpy()))

@torchsnooper.snoop()
def annotate_to_xml(xml_file, img_path, boxes):
    annotation = ET.Element("annotation")
    annotate_img_info(annotation, img_path)

    cls_tensor = boxes.cls
    boxes_tensor = boxes.xyxy
    for box_tensor, cls in zip(boxes_tensor, cls_tensor):
        annotate_a_box(annotation, box_tensor, cls)

    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")

    with open(xml_file, "w") as f:
        f.write(xml_str)


@torchsnooper.snoop()
def annotate_one_image(img_path):
    # img_path = "02.jpg"

    model = YOLO("best.pt") # 将文件替换为你需要的模型权重文件
    results = model([img_path])
    boxes = results[0].boxes

    image_name = os.path.basename(img_path)
    xml_name = os.path.splitext(image_name)[-2]
    annotate_to_xml(f"{xml_name}.xml", img_path, boxes)

    return 0


if __name__ == "__main__":
    img_path = "02.jpg"  # 更换为你的图片路径
    annotate_one_image(img_path)


