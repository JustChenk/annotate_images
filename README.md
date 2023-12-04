# annotate_images
 Automatic labeling ：使用YOLOv8自动标注数据集

# xml文件路径
所生成的xml路径与该脚本文件路径相同；  
可通过linux 命令: "mv *.xml  file_xml"将生成的xml文件转移至指定文件夹（file_xml）；  
然后将该文件夹与图片文件夹放于同一目录

# 使用教程
## 两个python文件需在同一目录下
result_to_xml_demo.py  单张图片效果展示  
result_to_xml_batch.py 批量处理单个文件夹内所有图片

## 在result_to_xml_demo.py 文件中：
### 修改模型权重文件
找到如下函数

```python
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
```

将Yolov8 模型权重修改为针对所要标记数据集的权重文件
### 修改所要标记的类型

在如下代码中，将class对应的标签类别根据你所要标注的数据集进行修改
```python
def annotate_a_box(annotation, box_tensor, cls_tensor):
    object_elem = ET.SubElement(annotation, "object")

    # 判断这个box的class
    # sub_text_ele(object_elem, "name", "human")
    if int(int(cls_tensor.item())) == 1:
        sub_text_ele(object_elem, "name", "human")
    elif int(int(cls_tensor.item())) == 0:
        sub_text_ele(object_elem, "name", "face")
    else:
        sub_text_ele(object_elem, "name", "others")

    sub_text_ele(object_elem, "pose", "Unspecified")
    sub_text_ele(object_elem, "truncated", "0")
    sub_text_ele(object_elem, "difficult", "0")

    bndbox = ET.SubElement(object_elem, "bndbox")
    for i, _name in enumerate(["xmin", "ymin", "xmax", 'ymax']):
        sub_text_ele(bndbox, _name, str(box_tensor[i].numpy()))
```

## 单张图片demo展示
于result_to_xml_demo.py 文件中，将图片路径修改为你本地图片路径

## 在result_to_xml_batch.py 文件中修改图片文件夹路径：
在如下函数中，输入你自己图片文件夹的路径 img_file.
```python
if __name__ == "__main__":
    img_file = r"Your\image\files\path"
    annotate_batch_image(img_file)
```
# 效果分析
1.模型边界尚未明确，后续仍需要人工优化  
2.经测试发现，同一目标会标注多次  
3.只能进行基础的分类，无法进行进一步的动作识别  
4.使用前需要准备部分数据集进行训练得到相应的训练权重  
5.仅仅适用于detection 

若没有相关权重文件<br>可以使用官方给的预训练权重




