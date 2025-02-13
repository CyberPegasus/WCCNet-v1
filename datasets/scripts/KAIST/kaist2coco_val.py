import json
import cv2 as cv
import os
import shutil
from tqdm import tqdm
import numpy as np

def del_file(filepath):
    """
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

# training settings following MLPD: https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection/
# Detailed information in https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection/blob/8e583929a97b9f959634efa69174435ad0252a57/src/config.py#L77
TRAIN_SET = {'hRng': (12, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}
def check_xywh(xmin:int,ymin:int,w:int,h:int,LOAD_SET:dict=TRAIN_SET):
    if  xmin < LOAD_SET['xRng'][0] or \
        ymin < LOAD_SET['yRng'][0] or \
        xmin+w > LOAD_SET['xRng'][1] or \
        ymin+h > LOAD_SET['yRng'][1] or \
        w < LOAD_SET['wRng'][0] or \
        w > LOAD_SET['wRng'][1] or \
        h < LOAD_SET['hRng'][0] or \
        h > LOAD_SET['hRng'][1]:        
        return False
    else:
        return True

# generate txt from kaist
annos_ori_path = 'datasets/kaist/annos/paired/'
train_split = 'datasets/kaist/split/val.txt'
modality = {'ir':'lwir','rgb':'visible'}
dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root_dir = os.path.join(dataset_dir,"kaist/")
write_path = root_dir+'annotations/'

if not os.path.exists(write_path):
    os.mkdir(write_path)

imgs_path = root_dir+'val/'
anno_path = root_dir+'annos/paired/'

one_class_mode = True
assert one_class_mode, 'one_class_mode remains to True. The reason lies that, following prior works, the categories in KAIST are all set to 0.'
json_save_name = write_path + 'instances_val.json' if one_class_mode else 'instances_val_multilabel.json'
class_id_dict = {'people':0,'person':0,'cyclist':0,'person?':0} if one_class_mode else {'people':0,'person':1,'cyclist':2,'person?':3}

json_dict = {
    "images": [],
    "type": "instances",
    "annotations": [],
    "categories": []
}
# 读取所有标签和data
with open(train_split,'r') as f:
    sample_list = f.readlines()
    sample_list = [i.strip() for i in sample_list]
print("training sample num: ", len(sample_list))

print("writing category....")
if one_class_mode:
    category = {
        'supercategory': 'human',
        'id': 0,  # 类别的ID，数字
        'name': 'people',  # 类别的名字，如汽车、飞机
    }
    json_dict['categories'].append(category)
else:
    for i in class_id_dict.keys():
        cate = i
        cid = class_id_dict[cate]
        print(f"class name:{cate},calss ID:{cid}")
        category = {
            'supercategory': 'human',
            'id': cid,  # 类别的ID，数字
            'name': cate,  # 类别的名字，如汽车、飞机
        }
        json_dict['categories'].append(category)
        
count = 1
bbox_id = 1
pbar = tqdm(sample_list)
for sample in pbar:
    pbar.set_description(f"{sample}")
    _img_name = sample.split('/')[-1]
    # anno path
    ir_anno_path = annos_ori_path+sample.split('/')[0]+'/'+sample.split('/')[1]+'/'+modality['ir']+'/'+_img_name+'.txt'
    rgb_anno_path = annos_ori_path+sample.split('/')[0]+'/'+sample.split('/')[1]+'/'+modality['rgb']+'/'+_img_name+'.txt'
    assert os.path.exists(ir_anno_path) and os.path.exists(rgb_anno_path), f'{ir_anno_path} or {rgb_anno_path} not exists'
    # img read
    ir_img_path = sample.split('/')[0]+'/'+sample.split('/')[1]+'/'+modality['ir']+'/'+_img_name+'.png'
    rgb_img_path = sample.split('/')[0]+'/'+sample.split('/')[1]+'/'+modality['rgb']+'/'+_img_name+'.png'
    assert os.path.exists(imgs_path+ir_img_path) and os.path.exists(imgs_path+rgb_img_path), f'{imgs_path+ir_anno_path} or {imgs_path+rgb_anno_path} not exists'
    
    # img to json
    ir_img = cv.imread(imgs_path+ir_img_path)
    rgb_img = cv.imread(imgs_path+rgb_img_path)
    assert ir_img.shape==rgb_img.shape
    height, width, channel = rgb_img.shape
    image = {
            'file_name': rgb_img_path,
            'file_name_ir': ir_img_path,
            'height': height,
            'width': width,
            'id': count
        }
    json_dict['images'].append(image)
    
    # anno to json
    with open(rgb_anno_path, "r") as f:
        rgb_txt = f.readlines()
        i = 0
        while i<len(rgb_txt):
            if 'unpaired' in rgb_txt[i] or 'bbGt' in rgb_txt[i]:
                rgb_txt.pop(i)
            else:
                rgb_txt[i] = rgb_txt[i].strip()
                i+=1
                
    with open(ir_anno_path,"r") as f:
        ir_txt = f.readlines()
        i = 0
        while i<len(ir_txt):
            if 'unpaired' in ir_txt[i] or 'bbGt' in ir_txt[i]:
                ir_txt.pop(i)
            else:
                ir_txt[i] = ir_txt[i].strip()
                i+=1
                
    assert len(rgb_txt)==len(ir_txt) and len(rgb_txt)>0, f'rgb len:{len(rgb_txt)} != ir len:{len(ir_txt)}'
    for index in range(len(rgb_txt)):
        rgb_line = rgb_txt[index].split()
        ir_line = ir_txt[index].split()
        assert rgb_line[0]==ir_line[0], f'{rgb_line[0]} != {ir_line[0]}'
        xmin = int((int(rgb_line[1]) + int(ir_line[1]))//2)
        xmax = int((int(rgb_line[1]) + int(rgb_line[3]) + int(ir_line[1]) + int(ir_line[3]))//2)
        ymin = int((int(rgb_line[2]) + int(ir_line[2]))//2)
        ymax = int((int(rgb_line[2]) + int(rgb_line[4]) + int(ir_line[2]) + int(ir_line[4]))//2)

        try:
            class_id = class_id_dict[rgb_line[0]]
        except:
            class_id = 0
            print(f'{rgb_line[0]} not in {class_id_dict.keys()}, and we set its classid to 0')
        b_w = abs(xmax-xmin)
        b_h = abs(ymax-ymin)
        b_area = b_w*b_h
        if check_xywh(xmin,ymin,b_w,b_h):
            annotation = {
                'image_id': count,
                'bbox': [xmin, ymin, b_w, b_h],
                'rgb_bbox': [int(rgb_line[1]), int(rgb_line[2]), int(rgb_line[3]), int(rgb_line[4])],
                'ir_bbox': [int(ir_line[1]), int(ir_line[2]), int(ir_line[3]), int(ir_line[4])],
                'category_id': class_id,  # 表示类别
                'id': bbox_id,
                'area': b_area,
                'iscrowd': 0
            }        
            json_dict['annotations'].append(annotation)
            bbox_id += 1
        else:
            continue
    count +=1
    
print("\n\nsaving json to "+json_save_name)
with open(json_save_name, 'w') as json_file:
    json_content = json.dumps(json_dict)
    json_file.write(json_content)
    json_file.close()

print("done.")