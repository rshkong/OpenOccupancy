import pickle
import copyreg

# Normal failure
try:
    pickle.dumps({1: 2}.keys())
    print("Normal: Success")
except Exception as e:
    print(f"Normal: Failed - {e}")

# Monkey patch
def pickle_dict_keys(k):
    return list, (list(k),)

copyreg.pickle(type({}.keys()), pickle_dict_keys)

try:
    res = pickle.dumps({1: 2}.keys())
    print("Patched: Success", type(pickle.loads(res)), pickle.loads(res))
except Exception as e:
    print(f"Patched: Failed - {e}")

