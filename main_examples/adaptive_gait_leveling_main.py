import os
from gait.body_leveling.legs_heights import NeuromorphicLegHeightModel
from gait.body_leveling.leveling_action import calculate_body_leveling_action
from gait.gait_impl import TripodMotion
from environment.hexapod_env import HexapodEnv
from gait.state_transitions import StageType

'''
Loading environment and environment
'''
BASE_DIR = os.path.dirname(__file__)
model_name = 'ramp'
xml_path = os.path.join(BASE_DIR, f'../mjcf_models/{model_name}.xml')
env = HexapodEnv(xml_path, frame_skip=300)
obs = env.reset()

'''
Init gaits and other simulation variables
'''
qpos_map = env.map_joint_qpos
gait = TripodMotion(qpos_map)

# sim_model = SimLegHeightModel(env)
for i in range(2):
    obs, reward, done, info = env.step({}, render=True)  # warmup

# init
neuro_model = NeuromorphicLegHeightModel(env)
obs, reward, done, info = env.step({}, callback=neuro_model.update, render=True)
while True:
    gait_state = gait.cycle.stages_cycle.curr
    action = gait.generate_action(obs)
    if StageType.LEVEL in gait_state:
        action = calculate_body_leveling_action(neuro_model, env.qpos, qpos_map)
        obs, reward, done, info = env.step(action, callback=neuro_model.update, frame_skip=100, render=True)
    else:
        obs, reward, done, info = env.step(action, callback=neuro_model.update, render=True)

        # rad_to_target = info['rad_to_target']
    # direction_manager.theta_change = rad_to_target
