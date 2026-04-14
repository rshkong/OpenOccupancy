from mmcv import Config
from mmdet3d.datasets import build_dataset
import projects.occ_plugin.datasets

cfg = Config.fromfile('projects/configs/baselines/CAM-R50_img1600_128x128x10.py')
print("Building config...")
# remove concat wrapper if exist 
dataset = build_dataset(cfg.data.train)

print(type(dataset))

def check_dict_keys(d, p):
    for k, v in d.items():
        if type(v).__name__ == "dict_keys":
            print(f"Found dict_keys at {p}[{k}]")
            return True
        elif isinstance(v, dict):
            if check_dict_keys(v, p + f"[{k}]"):
                return True
        elif isinstance(v, list) and len(v) > 0:
            if isinstance(v[0], dict):
                if check_dict_keys(v[0], p + f"[{k}][0]"):
                    return True
    return False

if hasattr(dataset, "datasets"):
    d = dataset.datasets[0]
else:
    d = dataset
    
for attr in dir(d):
    try:
        val = getattr(d, attr)
    except:
        continue
        
    if type(val).__name__ == "dict_keys":
        print(f"Found dict_keys at dataset.{attr}")
    elif isinstance(val, dict):
        check_dict_keys(val, f"dataset.{attr}")
    elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
        check_dict_keys(val[0], f"dataset.{attr}[0]")
    
    # check pipeline
    if attr == "pipeline":
        if hasattr(val, "transforms"):
            for i, t in enumerate(val.transforms):
                for ta in dir(t):
                    try:
                        tv = getattr(t, ta)
                        if type(tv).__name__ == "dict_keys":
                            print(f"Found dict_keys at pipeline.transforms[{i}].{ta}")
                    except:
                        pass
        
print("Check done")
