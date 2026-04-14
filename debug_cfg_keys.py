from mmcv import Config

cfg = Config.fromfile('projects/configs/baselines/CAM-R50_img1600_128x128x10.py')

def check_keys(obj, path=""):
    if type(obj).__name__ == 'dict_keys':
        print(f"Found dict_keys at {path}")
        return True
    
    if isinstance(obj, dict) or type(obj).__name__ in ('ConfigDict', 'Config'):
        for k, v in obj.items():
            if check_keys(v, path + f"['{k}']"): return True
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if check_keys(v, path + f"[{i}]"): return True
    return False

check_keys(cfg._cfg_dict, "cfg")
