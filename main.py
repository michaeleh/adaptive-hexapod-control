import os

import mujoco_py as mjc

from gait import Gait
from hexapod import Hexapod

BASE_DIR = os.path.dirname(__file__)

model_name = 'mk3'
xml_path = os.path.join(BASE_DIR, f'hexapod-models/{model_name}/{model_name}.xml')
mjc_model = mjc.load_model_from_path(xml_path)
hexapod = Hexapod(mjc_model)
gait = Gait()
while True:
    action = gait.generate_action()
    hexapod.step(action)
    hexapod.render()
