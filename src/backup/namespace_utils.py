import json

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def save_namespace(FLAGS, out_path):
    FLAGS_dict = vars(FLAGS)
    with open(out_path, 'w',encoding='utf-8') as fp:
        json.dump(FLAGS_dict, fp)
        
def load_namespace(in_path):
    with open(in_path, 'r',encoding='utf-8') as fp:
        FLAGS_dict = json.load(fp)
    return Bunch(FLAGS_dict)