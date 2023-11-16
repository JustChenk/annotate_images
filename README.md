# annotate_images
 Automatic labeling ：使用YOLOv8自动标注数据集

# xml文件路径
所生成的xml路径与该脚本文件路径相同；  
可通过linux 命令: "mv *.xml  file_xml"将生成的xml文件转移至指定文件夹（file_xml）；  
然后将该文件夹与图片文件夹放于统一路径

# 使用需知
## 在result_to_xml_demo.py 文件中修改模型权重：
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

将Yolov8 模型权重修改为你需要的权重文件

## 单张图片demo展示
于result_to_xml_demo.py 文件中，将图片路径修改为你本地图片路径即可

## 在result_to_xml_batch.py 文件中修改图片文件夹路径：
找到如下函数：
```python

```

将 改为你自己图片的文件夹路径



