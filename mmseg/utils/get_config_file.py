import os

def get_config_file_from_params(params):
    file_path = "./configs/{}".format(params["args"]["model"])
    
    if params["args"]["model"] == "pspnet":
        file_name = "{}_".format(file_name) + "{}".format(params["epochs"]) + '_openbayes.py'
        #file_name = "{}_1k_openbayes.py".format(file_name)
        file_path = os.path.join(file_path, file_name)
        print(file_path)
    return file_path
