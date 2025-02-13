import sys
import os
import json
import xml.etree.ElementTree as ET
import random

txt_read = ['datasets/alignedFLIR/align_train.txt','datasets/alignedFLIR/align_val.txt']
txt_write = ['datasets/alignedFLIR/xml_train.txt','datasets/alignedFLIR/xml_val.txt','datasets/alignedFLIR/xml_test.txt']
dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
anno_dir = 'datasets/alignedFLIR/annos/'
for index in range(len(txt_read)):
    with open(txt_read[index],'r') as fr:
        _lines = fr.readlines()
        _lines = [i.strip() for i in _lines]
        _lines = [i.split('_')[1] for i in _lines]
        _lines = [i+'.xml' for i in _lines]
        for i in _lines:
            assert os.path.exists(anno_dir+i), f'Not exists: {anno_dir+i}'
        _lines = [i+'\n' for i in _lines]
        if index==0:
            train_ratio = 0.8
            train_num = int(len(_lines)*train_ratio)
            train_lines = random.sample(_lines,train_num)
            val_lines = [i for i in _lines if i not in train_lines]
            with open(txt_write[0],'w') as fw:
                fw.writelines(train_lines)
                fw.close()
            with open(txt_write[1],'w') as fw:
                fw.writelines(val_lines)
                fw.close()
        else:
            with open(txt_write[2],'w') as fw:
                fw.writelines(_lines)
                fw.close()