# -*- coding:utf-8 -*-
"""数据增强
   1. 翻转变换 flip
   2. 随机修剪 random crop
   3. 色彩抖动 color jittering
   4. 平移变换 shift
   5. 尺度变换 scale
   6. 对比度变换 contrast
   7. 噪声扰动 noise
   8. 旋转变换/反射变换 Rotation/reflection
"""

from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
import logging

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


flag=1
filp_flag=1
i=0
class DataAugmentation:
    """
    包含数据增强的八种方式
    """

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def randomRotation(image, mode=Image.BICUBIC):
        global flag,random_angle
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        if(flag==3):
            random_angle = 90   
            flag=1                               
        elif(flag == 2):
            random_angle = 180
            flag=3
        elif(flag == 1):
            random_angle = 270
            flag=2
        return image.rotate(random_angle, mode)
        
    @staticmethod   
    def randomFlip(image):
        global filp_flag       
        if(filp_flag == 3):
            filp_flag=1
            return image
        elif(filp_flag == 2):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            filp_flag = 3
        elif(flag == 1):
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            filp_flag=2    
        return image
        
             
    @staticmethod
    def randomColor(image):
        '''
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        '''
        
        return image

    @staticmethod
    def saveImage(image, path):
        image.save(path)

def makeDir(path):
    try:
        if not os.path.exists(path):
            if not os.path.isfile(path):
                # os.mkdir(path)
                os.makedirs(path)
            return 0
        else:
            return 1
    except:
        #print str(e)
        return -2

def imageOps(func_name, image, des_path, file_name, times=3):
    funcMap = {"randomRotation":DataAugmentation.randomRotation,
                "randomColor": DataAugmentation.randomColor,              
               "randomFlip":DataAugmentation.randomFlip,
               }
    if funcMap.get(func_name) is None:
        logger.error("%s is not exist", func_name)
        return -1

    for _i in range(0, times, 1):
        new_image = funcMap[func_name](image)
        DataAugmentation.saveImage(new_image, os.path.join(des_path, func_name + str(_i) + file_name))



opsList = {"randomRotation","randomColor","randomFlip"}

def threadOPS(path, new_path):
    """
    多线程处理事务
    :param src_path: 资源文件
    :param des_path: 目的地文件
    :return:
    """
    if os.path.isdir(path):
        img_names = os.listdir(path)
    else:
        img_names = [path]
    for img_name in img_names:
        global i
        i+=1
        print ( "%r nd: "%i,img_name)
        tmp_img_name = os.path.join(path, img_name)
        if os.path.isdir(tmp_img_name):
            if makeDir(os.path.join(new_path, img_name)) != -1:
                threadOPS(tmp_img_name, os.path.join(new_path, img_name))
            else:
                print ('create new dir failure')
                return -1
                # os.removedirs(tmp_img_name)
        elif tmp_img_name.split('.')[1] != "DS_Store":
            # 读取文件并进行操作
            image = DataAugmentation.openImage(tmp_img_name)
            threadImage = [0] * 3
            _index = 0
            for ops_name in opsList:
                threadImage[_index] = threading.Thread(target=imageOps,
                                                       args=(ops_name, image, new_path, img_name,))
                threadImage[_index].start()
                _index += 1
                time.sleep(0.05)

if __name__ == '__main__':    
    #threadOPS("./data/train/auge_images","./auge_image_data")
    #masks 和 images 得分开做，因为mask的color都一样，而image的color要变
    threadOPS("./data/train/masks","./data/train/Auge_mask_data")
    
    
    
