"""
单张图片demo示例
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
    """
    return the shape of image
    Args:
        img_path:

    Returns:

    """
    img = cv2.imread(img_path)
    return img.shape


def sub_text_ele(parent_ele, ele_name, ele_text):
    """
    创建子
    Args:
        parent_ele: parent element
        ele_name: children element tag
        ele_text: children element dictionary of attributes

    Returns:

    """
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
    """

    Args:
        annotation:xml文件的root element
        cls_tensor: 一个cls张量

    Returns:

    """

    if int(int(cls_tensor.item())) == 1:
        sub_text_ele(annotation, "name", "human")
    else:
        sub_text_ele(annotation, "name", "face")


def annotate_a_box(annotation, box_tensor, cls_tensor):
    """

    Args:
        annotation: xml文件的root element
        box_tensor: 一个box的张量

    Returns:

    """
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
    """

    Args:
        xml_file:将生成的xml文件保存为xml_file
        img_path: 处理的图片路径
        boxes_tensor: Yolo处理结果，的.boxes属性

    Returns:

    """
    annotation = ET.Element("annotation")
    annotate_img_info(annotation, img_path)

    # boxes_cls = boxes.cls
    # 还是要吧class和box参数一起写
    # for cls in boxes_cls:
    #     annotate_a_class(annotation, cls)

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

    model = YOLO("human_and_face_best.pt") # 将文件替换为你需要的模型权重文件
    results = model([img_path])
    boxes = results[0].boxes

    image_name = os.path.basename(img_path)
    xml_name = os.path.splitext(image_name)[-2]
    annotate_to_xml(f"{xml_name}.xml", img_path, boxes)

    return 0


if __name__ == "__main__":
    img_path = "02.jpg"
    annotate_one_image(img_path)


