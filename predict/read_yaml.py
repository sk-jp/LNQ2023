from addict import Dict
import yaml


def read_yaml(fpath='./model.yaml'):
    with open(fpath, mode='r') as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

