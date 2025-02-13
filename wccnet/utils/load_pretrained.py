import os
import torch
import argparse, json
from wccnet.exp import get_exp

def load_pretrained(model=None, pretrain:str='darknet53'):
    assert pretrain in ['darknet53'], f'{pretrain} not implemented'
    pretrained_path = 'weights/yolox_darknet.pth'
    assert os.path.exists(pretrained_path), f"{pretrained_path} not exist."
    device = next(model.parameters()).device
    ckpt = torch.load(pretrained_path, map_location=device)
    with open("weights/out.txt", "w") as f:
        print(ckpt['model'].keys(),file= f)
        print(model.state_dict().keys(),file=f)

if __name__ == "__main__":
    cfg_file= ''
    # True for args input, False for dict load
    parser = argparse.ArgumentParser("WCCNet train parser")
    args= parser.parse_args()
    with open(cfg_file,'r') as f:
        args.__dict__=json.load(f)
    
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    model = exp.get_model()
    device = "cuda:{}".format(args.gpuid)
    model.to(device)
    model.train()
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)
    pass
    # load_pretrained(model=model)