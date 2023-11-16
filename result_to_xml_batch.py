from ultralytics import YOLO
import os
import sys
import torchsnooper
from result_to_xml_demo import annotate_one_image

"""
23/11/15
v1.0 所生成的xml路径与该脚本文件路径相同；
可通过linux 命令: "mv *.xml  file_xml"将生成的xml文件转移至指定文件夹；
然后将该文件夹与图片文件夹放于统一路径

下一步进行优化，将xml文件放置指定文件夹
"""

@torchsnooper.snoop()
def traverse_through_folders(img_file):
    """
    返回文件夹内所有文件（列表）
    Args:
        img_file:图片文件夹路径

    Returns:

    """
    imgs_name = os.listdir(img_file)
    imgs = [os.path.join(img_file, file_name) for file_name in imgs_name]
    return imgs


def annotate_batch_image(img_file):
    """
    获取所有图片的路径
    Args:
        img_file:图片文件夹路径

    Returns:

    """
    imgs = traverse_through_folders(img_file)

    for img in imgs:
        annotate_one_image(img)


if __name__ == "__main__":
    img_file = r"C:\github\pycode\practice\Examples\23_11\images"
    annotate_batch_image(img_file)