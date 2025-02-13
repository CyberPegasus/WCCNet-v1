#!/usr/bin/python

# pip install lxml

import sys
import os
import json
import xml.etree.ElementTree as ET
from copy import deepcopy
from tqdm import tqdm
START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {"person": 1, "car": 2, "bicycle": 3}
# If necessary, pre-define category and its id

def get(root, name):
    vars = root.findall(name)
    return vars

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.split('.')[0].split('_')[1]
        return filename
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))


def convert(xml_list, xml_dir, json_file):
    fp = open(xml_list, 'r')
    json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    # 获取ID的list
    # path_root = 'train2017/' if 'train' in xml_list else 'val2017/'
    path_root = '/JPEGImages/'
    # list_dir
    name_list = fp.readlines()
    name_list = [i.strip().split('.')[0] for i in name_list]
    name_list = sorted(name_list,reverse=False)
    id_set = {}
    for i in range(len(name_list)):
        id_set[name_list[i]]=i+1
    fp.close()
    fp = open(xml_list, 'r')
    list_fp = fp.readlines()
    list_fp = sorted(list_fp)
    for line in tqdm(list_fp):
        line = line.strip()
        #print("Processing %s"%(line))
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError('%d paths found in %s'%(len(path), line))
        ## The filename must be a number
        image_id = get_filename_as_int(filename)
        image_id = id_set[image_id]
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        RGB_name = filename.split('.')[0]+'_RGB.jpg'
        IR_name = filename.split('.')[0]+'_PreviewData.jpeg'
        image = {'file_name': RGB_name,
                 'file_name_ir':IR_name,
                'height': height,
                'width': width,
                'id':image_id
            }
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) #- 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) #- 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    fp.close()


if __name__ == '__main__':
    for mode in ['train','val','test']:
        xml_list = f'datasets/alignedFLIR/xml_{mode}.txt'
        anno_dir = 'datasets/alignedFLIR/annos/'
        output_json = f'datasets/alignedFLIR/annotations/alignedFLIR_{mode}.json'
        convert(xml_list,anno_dir,output_json)
