import os

from model import Model

BASE_DIR = os.path.dirname(__file__)

model_name = 'mk3'
path_join = os.path.join(BASE_DIR, f'hexapod-models/{model_name}/{model_name}.xml')
model = Model(xml_path=path_join)
model.visualize()
