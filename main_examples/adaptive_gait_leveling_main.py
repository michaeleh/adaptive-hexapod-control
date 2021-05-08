import os
import pickle

from tqdm import tqdm

from environment.leg import direction_manager
from gait.body_leveling.legs_heights import NeuromorphicLegHeightModel, SimLegHeightModel
from gait.body_leveling.leveling_action import calculate_body_leveling_action, targets
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
params = {}

for tau, transform in [(0.01, 10), (0.05, 50), (0.07, 80), (0.1, 100), (1, 1000)]:
    for n_neurons in [100, 1000, 10000]:
        if (tau, transform, n_neurons) == (0.07, 80, 1000):
            continue
        params[f'{n_neurons}_neurons_{tau}_tau'] = (n_neurons, tau, transform)
for dirname, inps in params.items():
    print(dirname, inps)
    obs = env.reset()
    env.dirname = dirname

    '''
    Init gaits and other simulation variables
    '''
    qpos_map = env.map_joint_qpos
    gait = TripodMotion(qpos_map)
    sim_model = SimLegHeightModel(env)
    for i in range(2):
        obs, reward, done, info = env.step({}, render=True)  # warmup
    # init
    neuro_model = NeuromorphicLegHeightModel(*inps, env)
    obs, reward, done, info = env.step({}, callback=neuro_model.update, render=True)

    for _ in tqdm(range(265)):

        gait_state = gait.cycle.stages_cycle.curr
        action = gait.generate_action(obs)

        if StageType.LEVEL in gait_state:
            action = calculate_body_leveling_action(neuro_model, env.qpos, qpos_map)
            obs, reward, done, info = env.step(action, callback=neuro_model.update, frame_skip=200, render=True)
        else:
            obs, reward, done, info = env.step(action, callback=neuro_model.update, render=False)

        rad_to_target = info['rad_to_target']
        direction_manager.theta_change = rad_to_target

    import os

    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    with open(f'{dirname}/sim_input.pkl', 'wb') as fp:
        pickle.dump(neuro_model.history, fp)
    with open(f'{dirname}/targets.pkl', 'wb') as fp:
        pickle.dump(targets, fp)
    with open(f'{dirname}/sim_xy.pkl', 'wb') as fp:
        pickle.dump(neuro_model.model.get_xy(), fp)
    #
    # for i, raster in enumerate(neuro_model.model.get_raster()):
    #     with open(f'raster{i}.pkl', 'wb') as fp:
    #         pickle.dump(raster, fp)
    #         del raster
