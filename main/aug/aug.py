# 1、导入模块
import os

import random
from PIL import Image
from  random import choice
import json
import numpy as np
import shutil

path = './data/coco/train2017/'  # coco训练集地址
ann_file = './data/coco/ann/person_keypoints_train2017.json'
cut_dir = './data/coco/cuttu/'  # 按 bbox裁剪后存储
new_dir = './data/coco/aug_train2017/'  # 抠图随机抽一张缩放后 在背景图中遮挡后 生成图片存放的路径


# 扣取图像后按照bbox剪裁人物，然后用剪裁后的人物进行遮挡
bbox_list = []
# zd_rate=1
backfiles = os.listdir(path)
for img in backfiles:  # 遍历文件夹中的图片
    files_output = os.listdir(cut_dir)
    sample_out = choice(files_output)
    picture_path = os.path.join(os.path.abspath(cut_dir), sample_out)
    image = Image.open(picture_path)
    w = image.size[0]  # 抠图宽
    h = image.size[1]  # 抠图高
    json_data = json.load(open(ann_file, 'r'))
    for i in range(0, len(json_data['images'])):
        if json_data['images'][i]['file_name'] == img:  # 根据图片名找到对应的json中的'images'
            imgID2 = json_data['images'][i]['id']
            for ann in json_data['annotations']:  # 根据image_id找到对应的annotation
                if ann['num_keypoints'] != 0 and ann['image_id'] == imgID2:
                    bbox1 = ann['bbox']
                    bbox_list.append(bbox1)  # 将一个图像中所有bbox数据存储到
    if len(bbox_list) > 0:
        bbox2 = random.sample(bbox_list, 1)
        # bbox2= random.sample(bbox_list, int(zd_rate*len(bbox_list))) #   随机抽取一定比例的bbox  进行遮挡
        for bbox in bbox2:  # 按照bbox数据依次对图中所有人物进行遮挡//按随机选取的图中一个人物的bbox数据遮挡
            x = bbox[0]
            y = bbox[1]
            w1 = bbox[2]  # 原图中人物宽高
            h1 = bbox[3]
            a = w1 / w  # 放缩到与图中人物等宽
            background = Image.open(path + img)  # 获取背景图片
            x1 = random.randint(int(x + w1 / 2), int(x + w1))
            y1 = random.randint(int(y), int(y + h1 / 2))  # 选定后1/4位置进行随机遮挡
            new_w = int(w * a)
            new_h = int(h * a)  # 放缩后抠图的大小
            x2 = int(x1 - new_w / 2)
            y2 = int(y1 - new_h / 2)

            out = image.resize((new_w, new_h), Image.ANTIALIAS)
            # out.save(suo_dir + sample_out)  # 缩放存放

            background.paste(out, (x2, y2), out)  # 把背景和图片粘贴在一起  background.paste(out, (0, 0), mask=out.split()[3]) 都可
            # print(w, h, w1, h1, new_w, new_h, x1, y1)
            background.save(new_dir + img)  # 所有人物遮挡后保存为最后的图片
            # bbox_list.clear()  # 对每一个图遮挡后要清空bbox_list 防止下一个图的bbox数据累积存储到bbox_list
            #  由于一张图中多个人物实例以此遮挡，需要这一个存一个直到遮完，每次遮挡会自动保存，下次会更新遮挡图，一张图中会生成多个遮挡图。
            # 若遮挡后图存放在data/bufentu1  里，每次都是从原图遮挡，所以只会产生一个人物遮挡，但是只会保存有人物遮挡的图，所以考虑在原图存放中遮挡。但是会出现多了遮挡图，所以考虑抽取一个人物bbox进行遮挡


            # imagename = [f for f in os.listdir(os.path.join(suo_dir))]
            # backfiles = os.listdir(path)
            # for name_index in range(0, len(imagename)):
            # 通过imgID 找到其所有instance
            # imgID = 0
        img_name = (os.path.splitext(sample_out)[0]) + ".jpg"
        for i in range(0, len(json_data['images'])):
            if json_data['images'][i]['file_name'] == img_name:  # 根据图片名找到对应的json中的'images'
                imgID3 = json_data['images'][i]['id']
                # imgIds_list.append(imgID1)
                for ann in json_data['annotations']:  # 根据image_id找到对应的annotation
                    if ann['image_id'] == imgID3:
                        # anns_list.append(ann)
                        # print(ann)
                        bbox3 = [x2, y2, new_w, new_h]
                        num_keypoints = ann['num_keypoints']
                        area = ann['area'] * a * a
                        iscrowd = ann['iscrowd']
                        segmentation = ann['segmentation']
                        b = ann['bbox']
                        keypoints = np.array(ann['keypoints'])
                        keypoints[::3] = (keypoints[::3] - b[0]) * a + x2
                        keypoints[1::3] = (keypoints[1::3] - b[1]) * a + y2
                        keypoints = np.array(keypoints[::]).tolist()
                        # print (keypoints)
        # for img_index in range(0, len(backfiles)):
        for i in range(0, len(json_data['images'])):
            # if json_data['images'][i]['file_name'] == backfiles[img_index]:  # 根据图片名找到对应的json中的'images'
            if json_data['images'][i]['file_name'] == img:
                imgID = json_data['images'][i]['id']
                data_anno = dict(
                    segmentation=segmentation,
                    num_keypoints=num_keypoints,
                    area=area,
                    iscrowd=iscrowd,
                    keypoints=keypoints,
                    image_id=imgID,
                    bbox=bbox3,
                    category_id=1, )
                # id=ann_img['id']
                # 标注信息列表————依次增加新的字段
                json_data['annotations'].append(data_anno)
        coco_sub = dict()
        coco_sub['info'] = json_data['info']
        coco_sub['licenses'] = json_data['licenses']
        coco_sub['images'] = json_data['images']
        coco_sub['type'] = 'instances'
        coco_sub['annotations'] = []
        # 插入annotation信息
        coco_sub['annotations'].extend(json_data['annotations'])
        coco_sub['categories'] = json_data['categories']
        # 自此所有该插入的数据就已经插入完毕啦٩( ๑╹ ꇴ╹)۶
        json.dump(coco_sub, open(ann_file, 'w'))  # 更新后存在原来json中，且每张图片增加一个抠图标签在annotations后边
        bbox_list.clear()  # 对每一个图遮挡后要清空bbox_list 防止下一个图的bbox数据累积存储到bbox_list

    elif len(bbox_list) == 0:
        shutil.copy(path + img, new_dir + img)
        # 可实现人物图和风景图遮挡后都存储到新路径




