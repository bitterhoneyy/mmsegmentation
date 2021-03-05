import os

def get_config_file_from_params(params):
    file_path = "./configs/{}".format(params["args"]["model"])
    print(params)
    if params["args"]["model"] == "pspnet":
        file_name = "pspnet_{}".format(params["resnet_depth"]) + "-d8_{}".format(params["image_width"]) + "x{}".format(params["image_height"])   
        file_name = "{}_1k_voc12aug.py".format(file_name)
        file_path = os.path.join(file_path, file_name)
    return file_path
