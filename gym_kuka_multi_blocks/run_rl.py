import pybullet as p
import pybullet_envs
import pybullet_data
import numpy as np

import ray
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.agents import ddpg
from ray.rllib.agents import dqn
from ray.rllib.agents import ppo

def env_creator_kuka_gym(renders=True):
    import gym
    import gym_kuka_multi_blocks
    return gym.make("KukaMultiBlocks-v0")

def init_ddpg(render):

    register_env("my_env", env_creator_kuka_bl)

    config = ddpg.DEFAULT_CONFIG.copy()
    config["num_workers"] = 3

    env = env_creator_kuka_bl(renders=render)

    agent = ddpg.DDPGAgent(config=config, env="my_env")
    # agent.restore("/Users/dgrebenyuk/ray_results/my_experiment/APEX_DDPG_KukaMultiBlocks-v0_0_2018-11-11_09-19-105785pfg0/checkpoint_55/checkpoint-55")
    return agent, env

# ---- dump

#-----------------------------------

operation = 'move_pick'
# operation = 'place'

def env_creator_kuka_bl(renders=False):
    import gym_kuka_multi_blocks.envs.kuka_multi_blocks_gym_env as e
    env = e.KukaMultiBlocksEnv(renders=renders,
                               numObjects=4,
                               isTest=10,
                               operation=operation,
                               constantVector=False,
                               blocksInObservation=True,  # F - e1, T - e2 or e3
                               sensing=False,
                               num_sectors=(16, 8))
    return env


def init_ppo(render):

    register_env("my_env", env_creator_kuka_bl)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 3

    env = env_creator_kuka_bl(renders=render)

    agent = ppo.PPOAgent(config=config, env="my_env")

    if operation == 'move_pick':
        pass
        # test == 0
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test0/PPO_KukaMultiBlocks-v0_0_2019-03-27_11-13-30nbdyzah7/checkpoint_300/checkpoint-300")
        # test == 3 close blocks without obs and reward
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test3_nan/PPO_KukaMultiBlocks-v0_0_2019-05-07_01-53-28pb9qko3w/checkpoint_160/checkpoint-160")
        # test == 3 blocks in obs without reward l = 0
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test3_L0/PPO_KukaMultiBlocks-v0_0_2019-06-05_04-21-493mf7szg5/checkpoint_420/checkpoint-420")
        # test == 3 blocks in obs with reward l = 1
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test3_L1/PPO_KukaMultiBlocks-v0_0_2019-06-04_09-33-20bftdy21e/checkpoint_380/checkpoint-380")
        # test == 3 blocks in obs and 1/2 reward l = 1/2
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test3_L05/PPO_KukaMultiBlocks-v0_0_2019-06-04_06-41-35sb357e86/checkpoint_500/checkpoint-500")
        # test == 3 all transferring weights. L = 0 -> 1/2 -> 1
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test3_L0051/PPO_KukaMultiBlocks-v0_0_2019-05-22_05-20-56eaerq7wv/checkpoint_780/checkpoint-780")
        # test == 6 rew i - 0 L = 0
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test6_L0/PPO_KukaMultiBlocks-v0_0_2019-05-31_04-03-10csxkhr85/checkpoint_160/checkpoint-160")
        # test == 6 rew i - 0 L = 1/2
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test6_L05/PPO_KukaMultiBlocks-v0_0_2019-06-03_06-13-20aewb6md9/checkpoint_320/checkpoint-320")
        # test == 6 rew i - 0 L = 1
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test6_L1/PPO_KukaMultiBlocks-v0_0_2019-05-30_06-24-46ocwsyz_5/checkpoint_380/checkpoint-380")
        # test == 6 rew i - 0 L = 2
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test6_L2/PPO_KukaMultiBlocks-v0_0_2019-05-30_07-15-35o5j_km46/checkpoint_400/checkpoint-400")
        # test == 7 L = 0
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test7_L0/PPO_KukaMultiBlocks-v0_0_2019-06-07_03-24-361fgge794/checkpoint_380/checkpoint-380")
        # test == 7 L = 1
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test7_L1/PPO_KukaMultiBlocks-v0_0_2019-06-14_06-39-5131ljiuiv/checkpoint_480/checkpoint-480")
        # test == 9 L = 0
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test9_L0/PPO_KukaMultiBlocks-v0_0_2019-06-10_06-43-35w_hez6dd/checkpoint_320/checkpoint-320")
        # test == 9 L = 1
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test9_L1/PPO_KukaMultiBlocks-v0_0_2019-06-17_05-24-29x3r81755/checkpoint_680/checkpoint-680")
        # test == 10 L = 0
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_L0/PPO_KukaMultiBlocks-v0_0_2019-06-11_09-12-33pdqcwk31/checkpoint_400/checkpoint-400")
        # test == 10 L = 1/4
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_L025/PPO_KukaMultiBlocks-v0_0_2019-06-18_05-20-39dbbinv3s/checkpoint_500/checkpoint-500")
        # test == 10 L = 1/2
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_L05/PPO_KukaMultiBlocks-v0_0_2019-06-17_09-21-23oxfsd34a/checkpoint_620/checkpoint-620")
        # test == 10 L = 1
        # old agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_L1/PPO_KukaMultiBlocks-v0_0_2019-06-11_06-40-33bal8v6r6/checkpoint_440/checkpoint-440")
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_L1/PPO_KukaMultiBlocks-v0_0_2019-06-17_07-26-46lsr9nat9/checkpoint_720/checkpoint-720")
        # test == 11 L = 0
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test11_L0/PPO_KukaMultiBlocks-v0_0_2019-06-12_05-15-4298pl0bw_/checkpoint_300/checkpoint-300")
        # test == 11 L = 1
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test11_L1/PPO_KukaMultiBlocks-v0_0_2019-06-12_03-54-16gim7q4et/checkpoint_340/checkpoint-340")
        # test 10 4 blocks L = 0
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_4bl_L0/PPO_KukaMultiBlocks-v0_0_2019-06-19_11-29-17ythlnk10/checkpoint_540/checkpoint-540")
        # test 10 4 blocks L = 1/36
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_4bl_L1_36/PPO_KukaMultiBlocks-v0_0_2019-06-26_12-12-19kkxamwlz/checkpoint_780/checkpoint-780")
        # test 10 4 blocks L = 1/25
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_4bl_L1_25/PPO_KukaMultiBlocks-v0_0_2019-06-26_10-59-17opjl66l_/checkpoint_560/checkpoint-560")
        # test 10 4 blocks L = 1/16
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_4bl_L1_16/PPO_KukaMultiBlocks-v0_0_2019-06-26_06-10-40v_kht3r2/checkpoint_740/checkpoint-740")
        # test 10 4 blocks L = 1/9
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_4bl_L1_9/PPO_KukaMultiBlocks-v0_0_2019-06-24_05-44-45q4si2sn5/checkpoint_560/checkpoint-560")
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_4bl_L1_9/PPO_KukaMultiBlocks-v0_0_2019-06-26_04-40-29in_48mnv/checkpoint_560/checkpoint-560")
        # test 10 4 blocks L = 1/8
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_4bl_L1_8/PPO_KukaMultiBlocks-v0_0_2019-06-19_10-46-45w9e05x8b/checkpoint_960/checkpoint-960")
        # test 10 5 blocks L = 0
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_5bl_L0/PPO_KukaMultiBlocks-v0_0_2019-06-19_04-22-275kg5sg8s/checkpoint_580/checkpoint-580")
        # test 10 5 blocks L = 1/16
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_5bl_L1_16/PPO_KukaMultiBlocks-v0_0_2019-06-20_04-49-061qzuzb75/checkpoint_920/checkpoint-920")
        # test 10 5 blocks L = 1/25
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_5bl_L1_25/PPO_KukaMultiBlocks-v0_0_2019-06-24_03-57-446tcxiu69/checkpoint_1060/checkpoint-1060")
        # exp 1
        # test 10 4 blocks L = 0
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_e1_4bl_L0/PPO_KukaMultiBlocks-v0_0_2019-06-27_07-30-077i_1glku/checkpoint_280/checkpoint-280")
        # text 10 4 blocks L = 1/36
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_e1_4bl_L1_36/PPO_KukaMultiBlocks-v0_0_2019-07-01_05-47-59g8g02v02/checkpoint_340/checkpoint-340")
        # test 10 4 blocks L = 1/25
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_e1_4bl_L1_25/PPO_KukaMultiBlocks-v0_0_2019-07-01_07-22-00qhydh6va/checkpoint_440/checkpoint-440")
        # test 10 4 blocks L = 1/16
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_e1_4bl_L1_16/PPO_KukaMultiBlocks-v0_0_2019-07-01_05-07-19twff5yx9/checkpoint_260/checkpoint-260")
        # test 10 sensing (16, 8) 4 blocks L = 0
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_s16_8_4bl_L0/PPO_KukaMultiBlocks-v0_0_2019-07-29_05-46-38y250zip7/checkpoint_1450/checkpoint-1450")
        # test 10 sensing (16, 8) 4 blocks L = 1/25
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_s16_8_4bl_L1_25/PPO_KukaMultiBlocks-v0_0_2019-07-10_03-19-29_j5vgc99/checkpoint_1720/checkpoint-1720")
        # test 10 sensing (16, 8) 4 blocks L = 1/36
        # agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_s16_8_4bl_L1_36/PPO_KukaMultiBlocks-v0_0_2019-07-11_04-09-11of7noaqc/checkpoint_2220/checkpoint-2220")
    elif operation == 'place':
        # test == 12 e2 L = 0
        # agent.restore("/Users/dgrebenyuk/Research/policies/place/test12_L0/PPO_KukaMultiBlocks-v0_0_2019-06-12_08-40-31lhabnnke/checkpoint_220/checkpoint-220")
        # test == 12 e2 L = 1
        # agent.restore("/Users/dgrebenyuk/Research/policies/place/test12_L1/PPO_KukaMultiBlocks-v0_0_2019-06-12_10-11-07stzkm6nb/checkpoint_360/checkpoint-360")
        # test == 12 e2 5 bl L = 0
        # agent.restore("/Users/dgrebenyuk/Research/policies/place/test12_5bl_L0/PPO_KukaMultiBlocks-v0_0_2019-06-25_10-10-56gxsemph4/checkpoint_360/checkpoint-360")
        # test == 12 e2 5 blocks L = 1/16
        # agent.restore("/Users/dgrebenyuk/Research/policies/place/test12_5bl_L0_16/PPO_KukaMultiBlocks-v0_0_2019-06-25_07-20-03ga1409qk/checkpoint_360/checkpoint-360")
        # test == 12 e1 5 blocks L = 0
        # agent.restore("/Users/dgrebenyuk/Research/policies/place/test12_e1_5bl_L0/PPO_KukaMultiBlocks-v0_0_2019-07-03_02-26-41y8thz8zy/checkpoint_360/checkpoint-360")
        # test == 12 e1 5 blocks L = 1/16
        # agent.restore("/Users/dgrebenyuk/Research/policies/place/test12_e1_5bl_L1_16/PPO_KukaMultiBlocks-v0_0_2019-07-03_03-10-18fv9sdiyu/checkpoint_420/checkpoint-420")
        # test 12 sensing (16, 8) 5 blocks L = 0
        ## old without the top block in obs
        # agent.restore("/Users/dgrebenyuk/Research/policies/place/test12_s16_8_5bl_L0/PPO_KukaMultiBlocks-v0_0_2019-07-15_03-57-01zwhscs6x/checkpoint_600/checkpoint-600")
        ## new with the top block in obs
        # agent.restore("/Users/dgrebenyuk/Research/policies/place/test12_s16_8_5bl_L0/PPO_KukaMultiBlocks-v0_0_2019-07-17_03-38-44d0nyfysk/checkpoint_1100/checkpoint-1100")
        # test 12 sensing (16, 8) 5 blocks L = 1 /16
        # agent.restore("/Users/dgrebenyuk/Research/policies/place/test12_s16_8_5bl_L1_16/PPO_KukaMultiBlocks-v0_0_2019-07-12_07-21-502xznjp14/checkpoint_620/checkpoint-620")
        # test 12 sensing (16, 8) 5 to 6 blocks L = 0
        # agent.restore("/Users/dgrebenyuk/Research/policies/place/test12_s16_8_5to6_bl_L0/PPO_KukaMultiBlocks-v0_0_2019-07-17_12-27-52af6_ro56/checkpoint_1260/checkpoint-1260")
        # test 13 sensing (16, 8) 5 blocks L = 0
        agent.restore("/Users/dgrebenyuk/Research/policies/place/test13_s16_8_5bl_L0/PPO_KukaMultiBlocks-v0_0_2019-07-22_04-32-20gtbab48l/checkpoint_400/checkpoint-400")
    elif operation == 'move':
        agent.restore("/Users/dgrebenyuk/ray_results/move/PPO_KukaMultiBlocks-v0_0_2019-04-09_02-24-40kihke9e8/checkpoint_40/checkpoint-40")
    elif operation == 'pick':
        agent.restore("/Users/dgrebenyuk/ray_results/pick/PPO_KukaMultiBlocks-v0_0_2019-04-10_07-30-536dh0eu86/checkpoint_180/checkpoint-180")
    else:
        raise NotImplementedError

    return agent, env


def test_kuka(run="PPO", iterations=1, render=True, scatter=False, stats=False, hist=False):

    if run == "PPO":
        agent, env = init_ppo(render)
    elif run == "DDPG":
        agent, env = init_ddpg(render)
    else:
        env = env_creator_kuka_bl(renders=True)

    success = []
    steps = []
    rwds = []
    dists = []
    s_dists = []
    for j in range(iterations):
        reward = 0.0
        obs = env.reset()
        done = False
        i = 0
        while not done:
            action = agent.compute_action(obs)
            obs, rew, done, info = env.step(action)
            # obs, rew, done, info = env.step([0, 0, 0, 0])
            # print("__________REWARD____________", rew, info)
            reward += rew
            i += 1

        steps.append(i)
        rwds.append(reward)
        dists.append(info['disturbance'])

        if reward >= 30:
            success.append(1)
            s_dists.append(info['disturbance'])
        else:
            success.append(0)

        if j % 100 == 0 and j > 0:
            print('iteration: ', j)

    if scatter:
        import pandas as pd
        data = pd.DataFrame(dict(steps=steps, reward=rwds, success=success))
        print_scatter(data)

    if stats:
        import numpy as np
        import scipy.stats as st
        a = st.t.interval(0.95, len(success) - 1, loc=np.mean(success), scale=st.sem(success))
        b = st.t.interval(0.95, len(steps) - 1, loc=np.mean(steps), scale=st.sem(steps))
        c = st.t.interval(0.95, len(dists) - 1, loc=np.mean(dists), scale=st.sem(dists))
        d = st.t.interval(0.95, len(s_dists) - 1, loc=np.mean(s_dists), scale=st.sem(s_dists))
        print('Success rate:', sum(success) / iterations, '+-', sum(success) / iterations - a[0],  "conf int: ", a,
              '\nAverage time: ', sum(steps) / len(steps), '+-', sum(steps) / len(steps) - b[0], 'conf int: ', b,
              '\nAverage disturbance: ', sum(dists) / len(dists), '+-', sum(dists) / len(dists) - c[0], 'conf int', c,
              '\nSuccess disturbance: ', sum(s_dists) / len(s_dists), '+-', sum(s_dists) / len(s_dists) - d[0], 'conf int', d
              )

    if hist:
        print_hist(s_dists)

    return sum(success) / iterations, sum(steps) / len(steps)


def print_scatter(data):

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.lmplot('steps', 'reward', data=data, hue='success', fit_reg=False, palette=['r', 'g'], legend=False)
    plt.title('Reward vs Policy Length')
    plt.legend(title='Grasp Success', loc='lower left', labels=['False', 'True'])
    plt.xlabel('Policy Length (time steps)')
    plt.ylabel('Reward')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, -70, 70))
    plt.show()


def print_hist(data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.distplot(data, kde=False)
    plt.xlabel('Distance')
    plt.ylabel('Simulations')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, y2))
    plt.show()


ray.init()

test_kuka("DDPG", iterations=1, render=True, scatter=False, stats=False, hist=False)
# test_kuka(iterations=2000, render=False, scatter=True, stats=True, hist=True)


# case 3 L = 1
# 2000 iterations
# 0.6305 11.5405
# Success rate: (0.6093283602030073, 0.6516716397969926) Average time (10.972596439963114, 12.108403560036885)
#Success rate: 0.645 Average time:  11.366
#Success rate conf int:  (0.6240106609210494, 0.6659893390789506) Average time conf int:  (10.808356851366723, 11.923643148633277)

#Success rate: 0.639 conf int:  (0.6179327058186288, 0.6600672941813712)
#Average time:  11.629 conf int:  (11.060201038721187, 12.197798961278812)
#Average disturbance:  0.010888091603217836 conf int (0.009550386219497945, 0.012225796986937657)

# Success rate: 0.64 +- 0.021054550166084596 conf int:  (0.6189454498339154, 0.6610545501660846)
# Average time:  11.559 +- 0.567057910793995 conf int:  (10.991942089206004, 12.126057910793994)
# Average disturbance:  0.01073506107821092 +- 0.0011760357935814824 conf int (0.009559025284629437, 0.011911096871792402)
# Success disturbance:  0.007802613373576656 +- 0.0010148226334684655 conf int (0.00678779074010819, 0.008817436007045138)

# Success rate: 0.856 +- 0.015400075786906053 conf int:  (0.8405999242130939, 0.871400075786906)
# Average time:  4.6495 +- 0.08509603601297311 conf int:  (4.564403963987027, 4.734596036012973)
# Average disturbance:  0.009797464258425639 +- 0.0015071838045662388 conf int (0.0082902804538594, 0.01130464806299192)
# Success disturbance:  0.004181725719840956 +- 0.0006367220702145418 conf int (0.003545003649626414, 0.004818447790055515)

# case 2 L = 0
# 2000 iterations
#Success rate: 0.8145 Average time:  6.2235
#Success rate conf int:  (0.7974500844980298, 0.8315499155019702) Average time conf int:  (6.010670678903822, 6.436329321096177)
#Success rate: 0.8415 Average time:  6.122
#Success rate conf int:  (0.8254805934561529, 0.8575194065438472) Average time conf int:  (5.914806985417463, 6.329193014582537)

#Success rate: 0.821 conf int:  (0.8041847695449676, 0.8378152304550323)
#Average time:  6.0405 conf int:  (5.846213953495339, 6.23478604650466)
#Average disturbance:  0.009747560266182146 conf int (0.008264487929570552, 0.01123063260279378)

# Success rate: 0.817 +- 0.016960603553763387 conf int:  (0.8000393964462366, 0.8339606035537633)
# Average time: 5.9895 + - 0.19696383977996224 conf int: (5.792536160220037, 6.186463839779962)
# Average disturbance: 0.009151896237212239 + - 0.0013354974818864697 conf int(0.00781639875532577, 0.010487393719098698)
# Success disturbance: 0.008728120612915266 + - 0.0015765197285228221 conf int(0.007151600884392444, 0.010304640341438092)

# Success rate: 0.892 +- 0.013614416562135023 conf int:  (0.878385583437865, 0.905614416562135)
# Average time:  4.7995 +- 0.09489461407054645 conf int:  (4.704605385929454, 4.8943946140705465)
# Average disturbance:  0.0110960775752653 +- 0.0016804815494566878 conf int (0.009415596025808612, 0.01277655912472197)
# Success disturbance:  0.011224091884342032 +- 0.001857348409195778 conf int (0.009366743475146254, 0.013081440293537827)

# case 4 L = 0.5
#Success rate: 0.774 conf int:  (0.7556545052029937, 0.7923454947970063)
#Average time:  10.01 conf int:  (9.675478510097705, 10.344521489902295)
#Average disturbance:  0.007774296118416455 conf int (0.0068826405667399004, 0.008665951670093)

# Success rate: 0.795 +- 0.017707825130870547 conf int:  (0.7772921748691295, 0.8127078251308706)
# Average time:  9.794 +- 0.33346343966844216 conf int:  (9.460536560331558, 10.127463439668443)
# Average disturbance:  0.00700999283412813 +- 0.0007365547623445932 conf int (0.006273438071783537, 0.0077465475964727075)
# Success disturbance:  0.005583412979158749 +- 0.0007677455716534097 conf int (0.004815667407505339, 0.006351158550812138)

# Success rate: 0.8595 +- 0.01524283794178094 conf int:  (0.8442571620582191, 0.874742837941781)
# Average time:  6.1715 +- 0.13922191219031532 conf int:  (6.032278087809685, 6.310721912190315)
# Average disturbance:  0.009955514157438072 +- 0.000915850890198083 conf int (0.009039663267239989, 0.010871365047636155)
# Success disturbance:  0.007321887814081879 +- 0.0008798819999364226 conf int (0.0064420058141454566, 0.00820176981401828)

# case L = 0 -> 0.5 -> 1
# Success rate: 0.8285 conf int:  (0.8119658052666575, 0.8450341947333425)
# Average time:  6.4755 conf int:  (6.178700276437136, 6.772299723562864)
# Average disturbance:  0.009037177415832159 conf int (0.007669490760663391, 0.010404864071000958)
# Success rate: 0.8345 +- 0.01630109953602943 conf int:  (0.8181989004639706, 0.8508010995360294)
# Average time:  6.3965 +- 0.28708844469875405 conf int:  (6.109411555301246, 6.683588444698754)
# Average disturbance:  0.008965544381351715 +- 0.0013416962285999028 conf int (0.007623848152751812, 0.010307240609951562)
# Success disturbance:  0.008711868336400022 +- 0.0015533350258837373 conf int (0.0071585333105162845, 0.010265203362283735)

# fixed reward L = 0
# Success rate: 0.9875 +- 0.004873357688679247 conf int:  (0.9826266423113208, 0.9923733576886793)
# Average time:  4.7445 +- 0.10094567768373164 conf int:  (4.643554322316269, 4.845445677683732)
# Average disturbance:  0.10927323869610352 +- 0.004612431027111674 conf int (0.10466080766899184, 0.11388566972321505)
# Success disturbance:  0.11026429975364031 +- 0.004646916254722064 conf int (0.10561738349891825, 0.11491121600836215)

# Success rate: 0.9895 +- 0.004471026855712101 conf int:  (0.985028973144288, 0.9939710268557121)
# Average time:  4.708 +- 0.10314707776449161 conf int:  (4.604852922235509, 4.811147077764492)
# Average disturbance:  0.10996485138598601 +- 0.004616222708283171 conf int (0.10534862867770284, 0.1145810740942689)
# Success disturbance:  0.11059045464660476 +- 0.004645355324773365 conf int (0.1059450993218314, 0.11523580997137788)

# L = 0.5
# Success rate: 0.973 +- 0.007109565045234967 conf int:  (0.965890434954765, 0.9801095650452349)
# Average time:  9.991 +- 0.15062496997943065 conf int:  (9.840375030020569, 10.14162496997943)
# Average disturbance:  0.10315014970117425 +- 0.004193110751721588 conf int (0.09895703894945267, 0.10734326045289612)
# Success disturbance:  0.10316582346443075 +- 0.004278332786167266 conf int (0.09888749067826348, 0.10744415625059832)

# Success rate: 0.998 +- 0.0019596792682652353 conf int:  (0.9960403207317348, 0.9999596792682652)
# Average time:  3.318 +- 0.030825752755185842 conf int:  (3.287174247244814, 3.348825752755186)
# Average disturbance:  0.11162440433454432 +- 0.004623480967240731 conf int (0.10700092336730359, 0.11624788530178522)
# Success disturbance:  0.1117460914454765 +- 0.004630933773717996 conf int (0.1071151576717585, 0.11637702521919466)

# fixed reward L = 1
# Success rate: 0.9845 +- 0.005418492031546429 conf int:  (0.9790815079684536, 0.9899184920315465)
# Average time:  5.8795 +- 0.10451133666330215 conf int:  (5.774988663336698, 5.984011336663302)
# Average disturbance:  0.02575745734106577 +- 0.0010821591763364759 conf int (0.024675298164729294, 0.02683961651740221)
# Success disturbance:  0.02441525163562251 +- 0.0008397458266513683 conf int (0.02357550580897114, 0.025254997462273897)

# Success rate: 0.968 +- 0.007720001727564374 conf int:  (0.9602799982724356, 0.9757200017275643)
# Average time:  5.1355 +- 0.08875361702591711 conf int:  (5.046746382974083, 5.2242536170259175)
# Average disturbance:  0.02587204576215371 +- 0.0012002648205875106 conf int (0.0246717809415662, 0.027072310582741222)
# Success disturbance:  0.023395912484531076 +- 0.0008226352122324837 conf int (0.022573277272298593, 0.024218547696763525)

# fixed reward L = 2
# Success rate: 0.9055 +- 0.012831128705519168 conf int:  (0.8926688712944808, 0.9183311287055191)
# Average time:  7.7905 +- 0.2802080640118927 conf int:  (7.510291935988107, 8.070708064011892)
# Average disturbance:  0.029881164382556263 +- 0.0018732483113756172 conf int (0.028007916071180645, 0.0317544126939318)
# Success disturbance:  0.022906489371660153 +- 0.0011525962864420633 conf int (0.02175389308521809, 0.024059085658102154)

# test 7 L = 0 (3 blocks in contact)
# Success rate: 0.962 +- 0.008386563970953387 conf int:  (0.9536134360290466, 0.9703865639709534)
# Average time:  6.65 +- 0.1279999518703665 conf int:  (6.522000048129634, 6.777999951870367)
# Average disturbance:  0.024272107593939257 +- 0.0008309894056288229 conf int (0.023441118188310434, 0.02510309699956806)
# Success disturbance:  0.023382102185293744 +- 0.0007267658942966167 conf int (0.022655336290997127, 0.024108868079590333)

# test 7  L = 1
# Success rate: 1.0 +- nan conf int:  (nan, nan)
# Average time:  4.294 +- 0.05502333824380212 conf int:  (4.2389766617561975, 4.349023338243802)
# Average disturbance:  0.013327663382497438 +- 0.0006178176805859362 conf int (0.012709845701911502, 0.013945481063083416)
# Success disturbance:  0.013327663382497438 +- 0.0006178176805859362 conf int (0.012709845701911502, 0.013945481063083416)

# reward fixed
# Success rate: 0.998 +- 0.0019596792682652353 conf int:  (0.9960403207317348, 0.9999596792682652)
# Average time:  8.4655 +- 0.19338749738252403 conf int:  (8.272112502617476, 8.658887497382525)
# Average disturbance:  0.008095554740937332 +- 0.0007466683427928798 conf int (0.007348886398144452, 0.008842223083730187)
# Success disturbance:  0.007760048809731218 +- 0.0006475727909786271 conf int (0.007112476018752591, 0.00840762160070983)

# test 7 tested on test 8 L = 0
# Success rate: 0.3675 +- 0.021147725585974275 conf int:  (0.3463522744140257, 0.38864772558597427)
# Average time:  10.83 +- 0.4098777857536273 conf int:  (10.420122214246373, 11.239877785753627)
# Average disturbance:  0.05681380045839736 +- 0.0028661044444087194 conf int (0.05394769601398864, 0.059679904902806204)
# Success disturbance:  0.041359229215803664 +- 0.003227374986316957 conf int (0.03813185422948671, 0.044586604202120704)

# test 7 tested on test 8 L = 1
# Success rate: 0.175 +- 0.016666742602595208 conf int:  (0.15833325739740478, 0.1916667426025952)
# Average time:  15.1885 +- 0.7255940310606661 conf int:  (14.462905968939333, 15.914094031060666)
# Average disturbance:  0.06144503810191556 +- 0.0033067767040693835 conf int (0.058138261397846176, 0.06475181480598552)
# Success disturbance:  0.06641681642407944 +- 0.005459678908043809 conf int (0.06095713751603563, 0.07187649533212326)

# fixed reward
# Success rate: 0.211 +- 0.017897173940033845 conf int:  (0.19310282605996615, 0.22889717394003384)
# Average time:  7.614 +- 0.25491536008638693 conf int:  (7.359084639913613, 7.868915360086387)
# Average disturbance:  0.28914447470867 +- 0.010483135035701419 conf int (0.2786613396729686, 0.2996276097443705)
# Success disturbance:  0.07171597813221874 +- 0.0045364078826586746 conf int (0.06717957024956006, 0.07625238601487735)

# test 9 L = 0 row + T
# Success rate: 0.853 +- 0.015532376625155497 conf int:  (0.8374676233748445, 0.8685323766251555)
# Average time:  5.424 +- 0.1284330556278359 conf int:  (5.2955669443721645, 5.552433055627836)
# Average disturbance:  0.15205380835491858 +- 0.004105681161370056 conf int (0.14794812719354852, 0.15615948951628825)
# Success disturbance:  0.1720923325884254 +- 0.0040232944945668425 conf int (0.16806903809385856, 0.17611562708299186)

# test 9 L = 1
# Success rate: 0.5985 +- 0.02150203586333732 conf int:  (0.5769979641366627, 0.6200020358633374)
# Average time:  6.421 +- 0.24697904597147957 conf int:  (6.174020954028521, 6.66797904597148)
# Average disturbance:  0.054583708474906664 +- 0.002907915300417112 conf int (0.05167579317448955, 0.05749162377532383)
# Success disturbance:  0.03061425422129138 +- 0.0015596727384339024 conf int (0.02905458148285748, 0.03217392695972522)

# fixed reward
# Success rate: 0.3145 +- 0.020366611734546736 conf int:  (0.29413338826545327, 0.33486661173454674)
# Average time:  8.257 +- 0.12255122030288668 conf int:  (8.134448779697113, 8.379551220302886)
# Average disturbance:  0.1806398962505501 +- 0.00499635844833507 conf int (0.17564353780221503, 0.18563625469888512)
# Success disturbance:  0.048273847757194614 +- 0.003746653379454422 conf int (0.04452719437774019, 0.052020501136648994)

# test 10 L = 0 tower
# Success rate: 0.9175 +- 0.012067984409531496 conf int:  (0.9054320155904685, 0.9295679844095315)
# Average time:  9.337 +- 0.2500090504457049 conf int:  (9.086990949554295, 9.587009050445705)
# Average disturbance:  0.019517770170799534 +- 0.001177092390106653 conf int (0.01834067778069288, 0.020694862560906118)
# Success disturbance:  0.019144198393084545 +- 0.0012120836793179726 conf int (0.017932114713766573, 0.020356282072402455)

# fixed reward 1+2
# Success rate: 0.937 +- 0.010657241318267507 conf int:  (0.9263427586817325, 0.9476572413182676)
# Average time:  9.5615 +- 0.24897485373701222 conf int:  (9.312525146262988, 9.810474853737013)
# Average disturbance:  0.34218321872188734 +- 0.02055915540449943 conf int (0.3216240633173879, 0.362742374126387)
# Success disturbance:  0.33522487553121344 +- 0.0211397415982591 conf int (0.31408513393295434, 0.35636461712947254)

# test 10 L = 0 all 3
# Success rate: 0.9475 +- 0.009783043637932476 conf int:  (0.9377169563620675, 0.9572830436379325)
# Average time:  4.749 +- 0.19103108412170133 conf int:  (4.557968915878298, 4.940031084121701)
# Average disturbance:  1.1409409744860946 +- 0.032637896474408734 conf int (1.1083030780116858, 1.1735788709605006)
# Success disturbance:  1.1724127802026585 +- 0.032729638361407076 conf int (1.1396831418412514, 1.2051424185640647)

# test 10 L = 1/4
# Success rate: 0.965 +- 0.00806125034540861 conf int:  (0.9569387496545914, 0.9730612503454086)
# Average time:  3.738 +- 0.046984415151478665 conf int:  (3.6910155848485213, 3.7849844151514787)
# Average disturbance:  0.18391771163562853 +- 0.00779407379842173 conf int (0.1761236378372068, 0.19171178543404943)
# Success disturbance:  0.17014771838651332 +- 0.006572473487856306 conf int (0.16357524489865702, 0.17672019187436874)

# test 10 L = 1/2
# Success rate: 0.708 +- 0.01994402382384053 conf int:  (0.6880559761761594, 0.7279440238238405)
# Average time:  7.623 +- 0.13373644479284152 conf int:  (7.489263555207159, 7.756736444792842)
# Average disturbance:  0.2848685804410963 +- 0.01618772717858402 conf int (0.2686808532625123, 0.30105630761968055)
# Success disturbance:  0.1223726590577333 +- 0.006110250151973043 conf int (0.11626240890576026, 0.12848290920970595)

# test 10 L = 1
# Success rate: 0.968 +- 0.007720001727564374 conf int:  (0.9602799982724356, 0.9757200017275643)
# Average time:  8.5535 +- 0.22020309572639185 conf int:  (8.333296904273608, 8.773703095726392)
# Average disturbance:  0.015483135043874621 +- 0.0008144897505024916 conf int (0.01466864529337213, 0.01629762479437715)
# Success disturbance:  0.015263042206100964 +- 0.0008266064389158122 conf int (0.014436435767185151, 0.016089648645016816)

# fixed - training failed

# test 10 4 blocks L = 0
# Success rate: 0.9465 +- 0.009870563115419784 conf int:  (0.9366294368845802, 0.9563705631154198)
# Average time:  4.78 +- 0.19216382545640265 conf int:  (4.587836174543598, 4.972163825456403)
# Average disturbance:  1.068940250929562 +- 0.032028311290586364 conf int (1.0369119396389757, 1.100968562220151)
# Success disturbance:  1.0967574468622603 +- 0.03220770272007356 conf int (1.0645497441421867, 1.1289651495823356)

# test 10 4 blocks L = 1/36 all 3
# Success rate: 0.915 +- 0.012232768010626227 conf int:  (0.9027672319893738, 0.9272327680106263)
# Average time:  9.4805 +- 0.37304121436615034 conf int:  (9.107458785633849, 9.85354121436615)
# Average disturbance:  1.0336932001852137 +- 0.03315345997735086 conf int (1.0005397402078628, 1.066846660162565)
# Success disturbance:  1.1004938200583818 +- 0.0333144309653175 conf int (1.0671793890930643, 1.1338082510237015)


# test 10 4 blocks L = 1/25 all 3
# Success rate: 0.645 +- 0.02098933907895062 conf int:  (0.6240106609210494, 0.6659893390789506)
# Average time:  6.6755 +- 0.5474232244580826 conf int:  (6.128076775541918, 7.222923224458083)
# Average disturbance:  0.8426065366881051 +- 0.0336547145084205 conf int (0.8089518221796846, 0.8762612511965254)
# Success disturbance:  0.9022115532860512 +- 0.04165712806164423 conf int (0.860554425224407, 0.9438686813476941)

# two top
# test 10 4 blocks L = 1/16
# Success rate: 0.8775 +- 0.014381240086103753 conf int:  (0.8631187599138962, 0.8918812400861037)
# Average time:  7.133 +- 0.44504918973419105 conf int:  (6.687950810265809, 7.578049189734191)
# Average disturbance:  0.9345680500527002 +- 0.034329823682160865 conf int (0.9002382263705393, 0.9688978737348592)
# Success disturbance:  0.9980008454224907 +- 0.035562503265633705 conf int (0.962438342156857, 1.033563348688119)

# all three
# Success rate: 0.6385 +- 0.021073629061554078 conf int:  (0.6174263709384459, 0.659573629061554)
# Average time:  5.7895 +- 0.49900019142162044 conf int:  (5.29049980857838, 6.288500191421621)
# Average disturbance:  1.0017183153301974 +- 0.035205021454695395 conf int (0.966513293875502, 1.0369233367848936)
# Success disturbance:  1.0117853791868545 +- 0.04177917332113612 conf int (0.9700062058657184, 1.0535645525079884)

# test 10 4 blocks L = 1/9
# Success rate: 0.892 +- 0.013614416562135023 conf int:  (0.878385583437865, 0.905614416562135)
# Average time:  6.7995 +- 0.4126734627126849 conf int:  (6.386826537287315, 7.212173462712685)
# Average disturbance:  0.8784028862946983 +- 0.03167379518001345 conf int (0.8467290911146849, 0.9100766814747093)
# Success disturbance:  0.9330783705015483 +- 0.03287480691045974 conf int (0.9002035635910886, 0.9659531774120063)

# test 10 4 blocks L = 1/8
# Success rate: 0.896 +- 0.013389840514418494 conf int:  (0.8826101594855815, 0.9093898405144185)
# Average time:  6.9905 +- 0.4284681467905287 conf int:  (6.562031853209471, 7.418968146790529)
# Average disturbance:  0.8092149053242786 +- 0.03305672735208298 conf int (0.7761581779721957, 0.8422716326763621)
# Success disturbance:  0.8595704664100413 +- 0.03446135542628248 conf int (0.8251091109837588, 0.8940318218363251)

# test 10 5 blocks L = 0
# Success rate: 0.9005 +- 0.013129800541739733 conf int:  (0.8873701994582602, 0.9136298005417397)
# Average time:  6.0415 +- 0.2653180586177859 conf int:  (5.776181941382214, 6.306818058617786)
# Average disturbance:  1.1616675331676336 +- 0.03940446251347307 conf int (1.1222630706541605, 1.2010719956811116)
# Success disturbance:  1.237542799124619 +- 0.04068681678191566 conf int (1.1968559823427034, 1.2782296159065378)

# test 10 5 blocks L = 1/25
# Success rate: 0.7785 +- 0.018214653056650265 conf int:  (0.7602853469433497, 0.7967146530566502)
# Average time:  9.5065 +- 0.508209053898474 conf int:  (8.998290946101527, 10.014709053898475)
# Average disturbance:  1.2088247242872843 +- 0.043901043878324275 conf int (1.16492368040896, 1.2527257681656063)
# Success disturbance:  1.2822711960037294 +- 0.04839263294942331 conf int (1.233878563054306, 1.3306638289531514)

# test 10 5 blocks L = 1/16
# Success rate: 0.7925 +- 0.017787438670075706 conf int:  (0.7747125613299243, 0.8102874386700757)
# Average time:  10.5405 +- 0.573691150633314 conf int:  (9.966808849366686, 11.114191150633314)
# Average disturbance:  1.251545389126918 +- 0.044347777607406824 conf int (1.207197611519511, 1.295893166734327)
# Success disturbance:  1.4256558045730872 +- 0.04701936417918895 conf int (1.3786364403938982, 1.4726751687522739)

# test 11 L = 0 five in "+"-shape
# Success rate: 0.9525 +- 0.009330051193269573 conf int:  (0.9431699488067304, 0.9618300511932696)
# Average time:  6.066 +- 0.15989874796822345 conf int:  (5.906101252031776, 6.225898747968223)
# Average disturbance:  0.027544623606721098 +- 0.0008376858120277136 conf int (0.026706937794693384, 0.028382309418748853)
# Success disturbance:  0.027242911966972257 +- 0.0008191977892421723 conf int (0.026423714177730085, 0.028062109756214485)

# test 11 L = 1
# Success rate: 0.9855 +- 0.005243448962607533 conf int:  (0.9802565510373925, 0.9907434489626076)
# Average time:  5.881 +- 0.12196017648534596 conf int:  (5.759039823514654, 6.002960176485346)
# Average disturbance:  0.022606506114785287 +- 0.0006424864231277666 conf int (0.02196401969165752, 0.02324899253791295)
# Success disturbance:  0.022508097958565516 +- 0.0006187166687316517 conf int (0.021889381289833865, 0.023126814627297043)

# place a block on the top of a tower of two
# test 12 2 blocks L = 0
# Success rate: 0.9915 +- 0.004026804560881891 conf int:  (0.9874731954391182, 0.9955268045608819)
# Average time:  5.8135 +- 0.03557332916178879 conf int:  (5.7779266708382115, 5.849073329161789)
# Average disturbance:  0.002909755369765412 +- 0.00019459882398648173 conf int (0.00271515654577893, 0.0031043541937518823)
# Success disturbance:  0.002879693305461727 +- 0.0001862915353621421 conf int (0.002693401770099585, 0.003065984840823857)

# test 12 5 blocks L = 0
# Success rate: 0.926 +- 0.011482225795244938 conf int:  (0.9145177742047551, 0.937482225795245)
# Average time:  7.7495 +- 0.3464110448498907 conf int:  (7.40308895515011, 8.095911044849892)
# Average disturbance:  0.0773274340080732 +- 0.017631923675353253 conf int (0.05969551033271994, 0.09495935768342642)
# Success disturbance:  0.009148544219457547 +- 0.0039477479336272525 conf int (0.005200796285830295, 0.013096292153084783)

# run again with magicaly better results
# Success rate: 0.988 +- 0.004776104549952143 conf int:  (0.9832238954500478, 0.9927761045499521)
# Average time:  5.5845 +- 0.04479377690072717 conf int:  (5.539706223099273, 5.629293776900727)
# Average disturbance:  0.006610366889490872 +- 0.0003782459101952197 conf int (0.006232120979295652, 0.006988612799686093)
# Success disturbance:  0.006537144256320775 +- 0.00037805894043759034 conf int (0.006159085315883185, 0.006915203196758371)

# test 12 5 blocks L = 1/16
# Success rate: 0.8615 +- 0.015151556767000307 conf int:  (0.8463484432329997, 0.8766515567670004)
# Average time:  12.617 +- 0.4163612318507983 conf int:  (12.200638768149203, 13.0333612318508)
# Average disturbance:  0.018088761115533543 +- 0.007004646223475535 conf int (0.011084114892058008, 0.025093407339009034)
# Success disturbance:  0.004441845091230491 +- 0.000413694673485324 conf int (0.004028150417745167, 0.004855539764715811)

# fixed re-run
# Success rate: 0.967 +- 0.00783564833725281 conf int:  (0.9591643516627472, 0.9748356483372528)
# Average time:  9.386 +- 0.049780967804920806 conf int:  (9.336219032195078, 9.43578096780492)
# Average disturbance:  0.003911337166330678 +- 0.00036275624036074833 conf int (0.00354858092596993, 0.00427409340669142)
# Success disturbance:  0.003946641931890971 +- 0.000373099234404207 conf int (0.0035735426974867644, 0.004319741166295171)

# test 12 L = 1
# Success rate: 0.9915 +- 0.004026804560881891 conf int:  (0.9874731954391182, 0.9955268045608819)
# Average time:  5.2675 +- 0.024380602655694972 conf int:  (5.243119397344305, 5.291880602655695)
# Average disturbance:  0.004160805352399772 +- 0.00024164431547268102 conf int (0.003919161036927091, 0.004402449667872449)
# Success disturbance:  0.0041583415772817045 +- 0.0002431950601718743 conf int (0.00391514651710983, 0.004401536637453575)

# test 12 5 bl on 4 bl L = 0
# Success rate: 0.036 +- 0.008171362873236679 conf int:  (0.02782863712676332, 0.044171362873236676)
# Average time:  28.5995 +- 0.7087069005929081 conf int:  (27.89079309940709, 29.308206900592907)
# Average disturbance:  0.39533127401350077 +- 0.026537104910398845 conf int (0.3687941691031019, 0.4218683789239003)
# Success disturbance:  0.015773132253401864 +- 0.007323619733430927 conf int (0.008449512519970937, 0.02309675198683279)

# test 12 5 bl on 4 bl L = 1/16
# Success rate: 0.0025 +- 0.0021904391155121226 conf int:  (0.00030956088448787743, 0.004690439115512123)
# Average time:  38.674 +- 0.26568744568984215 conf int:  (38.40831255431016, 38.93968744568984)
# Average disturbance:  0.273453095891218 +- 0.025310547811415923 conf int (0.2481425480798021, 0.298763643702637)
# Success disturbance:  0.0034559590277774338 +- 0.007295330752041767 conf int (-0.003839371724264333, 0.0107512897798192)

# exp 1
# test 10 e1 4 blocks L = 0
# Success rate: 0.928 +- 0.011338222258444608 conf int:  (0.9166617777415554, 0.9393382222584447)
# Average time:  5.042 +- 0.07718301824120033 conf int:  (4.9648169817587995, 5.1191830182412)
# Average disturbance:  1.3521075784250536 +- 0.02652980312939235 conf int (1.3255777752956612, 1.3786373815544437)
# Success disturbance:  1.342991060940879 +- 0.027308106356504602 conf int (1.3156829545843745, 1.3702991672973788)

# test 10 e1 4 blocks L = 1/36
# Success rate: 0.944 +- 0.010085205229926286 conf int:  (0.9339147947700737, 0.9540852052299262)
# Average time:  3.9535 +- 0.018781517458558383 conf int:  (3.9347184825414416, 3.9722815174585584)
# Average disturbance:  1.238680730422086 +- 0.0312397885563811 conf int (1.2074409418657048, 1.2699205189784675)
# Success disturbance:  1.2234607387632932 +- 0.032171745641969896 conf int (1.1912889931213233, 1.2556324844052649)

# test 10 e1 4 blocks L = 1/25
# Success rate: 0.4025 +- 0.021510803194712436 conf int:  (0.3809891968052876, 0.42401080319471246)
# Average time:  3.197 +- 0.2911343273977516 conf int:  (2.9058656726022485, 3.4881343273977516)
# Average disturbance:  0.9161613402383337 +- 0.031461859337622244 conf int (0.8846994809007115, 0.9476231995759558)
# Success disturbance:  0.8641833948542579 +- 0.05537937881567723 conf int (0.8088040160385807, 0.9195627736699349)

# test 10 e1 4 blocks L = 1/16
# Success rate: 0.4515 +- 0.021828400978333773 conf int:  (0.42967159902166624, 0.4733284009783338)
# Average time:  2.019 +- 0.03726187698324068 conf int:  (1.9817381230167594, 2.056261876983241)
# Average disturbance:  0.7417407807496608 +- 0.02975082172594823 conf int (0.7119899590237125, 0.771491602475607)
# Success disturbance:  0.5883750519489995 +- 0.04981469061199284 conf int (0.5385603613370067, 0.6381897425609926)

# test 12 e1 5 blocks L = 0
# Success rate: 0.905 +- 0.012861476304927177 conf int:  (0.8921385236950728, 0.9178614763049272)
# Average time:  7.69 +- 0.32430603933453206 conf int:  (7.365693960665468, 8.014306039334532)
# Average disturbance:  0.4366104552720279 +- 0.014949198723284374 conf int (0.42166125654874353, 0.45155965399531295)
# Success disturbance:  0.356515993452393 +- 0.0035584994057430985 conf int (0.3529574940466499, 0.36007449285813586)

# run again with magically better result
# Success rate: 0.9955 +- 0.0029358348316402827 conf int:  (0.9925641651683598, 0.9984358348316403)
# Average time:  5.848 +- 0.030131994379359206 conf int:  (5.817868005620641, 5.878131994379359)
# Average disturbance:  0.0062370588066286625 +- 0.00032329308795260147 conf int (0.005913765718676061, 0.006560351894581241)
# Success disturbance:  0.006213035622003578 +- 0.00032207604170796274 conf int (0.005890959580295616, 0.006535111663711519)

# test 12 e1 5 blocks L = 1/16
# Success rate: 0.866 +- 0.014942252633691977 conf int:  (0.851057747366308, 0.880942252633692)
# Average time:  7.895 +- 0.4770783557067215 conf int:  (7.417921644293278, 8.372078355706721)
# Average disturbance:  0.3266111923080376 +- 0.014381743901554167 conf int (0.3122294484064834, 0.3409929362095922)
# Success disturbance:  0.22830587822621515 +- 0.0015610371352397234 conf int (0.22674484109097542, 0.22986691536145493)

# fixed re-run
# Success rate: 0.974 +- 0.006980248470323636 conf int:  (0.9670197515296763, 0.9809802484703236)
# Average time:  4.2945 +- 0.0782910146939173 conf int:  (4.216208985306083, 4.3727910146939175)
# Average disturbance:  0.009273419794628902 +- 0.0006121642140785762 conf int (0.008661255580550326, 0.00988558400870751)
# Success disturbance:  0.00925188034589548 +- 0.0006236020674291239 conf int (0.008628278278466356, 0.009875482413324631)

# test 12 e1 5 bl on 4 bl L = 0
# Success rate: 0.035 +- 0.008061250345408624 conf int:  (0.02693874965459138, 0.04306125034540863)
# Average time:  39.075 +- 0.22935434035209568 conf int:  (38.84564565964791, 39.3043543403521)
# Average disturbance:  0.38734192475925494 +- 0.002338280553615568 conf int (0.3850036442056394, 0.38968020531286995)
# Success disturbance:  0.36745061607238 +- 0.009554168580313827 conf int (0.35789644749206617, 0.37700478465269416)

# test 12 e1 5 bl on 4 bl L = 1/16
# Fail

# test 10 sen (16, 8) 4 blocks L = 0
# Success rate: 0.9715 +- 0.007298751242296397 conf int:  (0.9642012487577036, 0.9787987512422964)
# Average time:  7.445 +- 0.1773414050065174 conf int:  (7.267658594993483, 7.622341405006518)
# Average disturbance:  0.6368948201939757 +- 0.03248625170962094 conf int (0.6044085684843548, 0.6693810719035942)
# Success disturbance:  0.6244996253123338 +- 0.032824587615548406 conf int (0.5916750376967854, 0.6573242129278798)

# test 10 sen (16, 8) 4 blocks L = 0
# re-trained
# Success rate: 0.994 +- 0.003387455097334624 conf int:  (0.9906125449026654, 0.9973874550973346)
# Average time:  4.3355 +- 0.07391954635469578 conf int:  (4.261580453645304, 4.4094195463546955)
# Average disturbance:  0.9912042232404342 +- 0.03388812106024808 conf int (0.9573161021801861, 1.0250923443006819)
# Success disturbance:  0.9894163920147926 +- 0.03403910658009934 conf int (0.9553772854346932, 1.0234554985948916)

# test 10 sen (16, 8) 4 on 3 blocks L = 0
# Success rate: 0.8295 +- 0.016495865866269277 conf int:  (0.8130041341337307, 0.8459958658662693)
# Average time:  12.474 +- 0.47761715019606577 conf int:  (11.996382849803934, 12.951617150196066)
# Average disturbance:  0.28318913817486696 +- 0.019683116375936116 conf int (0.26350602179893085, 0.3028722545508034)
# Success disturbance:  0.3030882627909042 +- 0.022133084306836526 conf int (0.28095517848406765, 0.3252213470977415)

# test 10 sen (16, 8) 4 on 3 blocks L = 0
# re-trained
# Success rate: 0.9545 +- 0.00914109861972734 conf int:  (0.9453589013802727, 0.9636410986197274)
# Average time:  6.3915 +- 0.31763527895503874 conf int:  (6.073864721044961, 6.7091352789550385)
# Average disturbance:  0.5197432877960895 +- 0.02612449035398262 conf int (0.49361879744210685, 0.5458677781500708)
# Success disturbance:  0.5336910921274732 +- 0.02686838665812874 conf int (0.5068227054693445, 0.5605594787856004)

# test 10 sen (16, 8) 4 blocks L = 1/25
# Success rate: 0.891 +- 0.013669632239438623 conf int:  (0.8773303677605614, 0.9046696322394386)
# Average time:  4.159 +- 0.09038166828141225 conf int:  (4.0686183317185876, 4.249381668281412)
# Average disturbance:  1.0501043309519298 +- 0.03416065661205803 conf int (1.0159436743398718, 1.08426498756399)
# Success disturbance:  1.044650408253222 +- 0.03626601100049309 conf int (1.0083843972527289, 1.0809164192537155)

# test 10 sen (16, 8) 4 blocks L = 1/36
# Success rate: 0.7585 +- 0.018773320295918716 conf int:  (0.7397266797040812, 0.7772733202959187)
# Average time:  9.8945 +- 0.41230747077154106 conf int:  (9.48219252922846, 10.306807470771542)
# Average disturbance:  0.6407982583695024 +- 0.032755913819562 conf int (0.6080423445499404, 0.6735541721890658)
# Success disturbance:  0.6700115251848306 +- 0.03837727973054794 conf int (0.6316342454542827, 0.7083888049153801)

# test 10 sen (16, 8) 4 on 3 blocks L = 1/25
# Success rate: 0.7165 +- 0.01976921199955317 conf int:  (0.6967307880004469, 0.7362692119995532)
# Average time:  4.189 +- 0.08550912661690102 conf int:  (4.103490873383099, 4.274509126616901)
# Average disturbance:  0.5665168598282163 +- 0.022659780555494557 conf int (0.5438570792727218, 0.5891766403837115)
# Success disturbance:  0.5956924145396831 +- 0.02821200658700218 conf int (0.567480407952681, 0.6239044211266846)

# test 10 sen (16, 8) 4 on row t7 L = 0
# Success rate: 0.7675 +- 0.018529145874210062 conf int:  (0.7489708541257899, 0.78602914587421)
# Average time:  10.3915 +- 0.44867434724150534 conf int:  (9.942825652758495, 10.840174347241506)
# Average disturbance:  0.25617523169149164 +- 0.010960274602593856 conf int (0.2452149570888978, 0.26713550629408583)
# Success disturbance:  0.28002997425409876 +- 0.012982077176250828 conf int (0.26704789707784793, 0.2930120514303509)

# test 10 sen (16, 8) 4 on 5 blocks L = 0
# Success rate: 0.766 +- 0.018570647360755532 conf int:  (0.7474293526392445, 0.7845706473607555)
# Average time:  15.7885 +- 0.6094186613051811 conf int:  (15.17908133869482, 16.397918661305184)
# Average disturbance:  1.0557150062002998 +- 0.044278292231443084 conf int (1.0114367139688567, 1.0999932984317407)
# Success disturbance:  1.2355828942314553 +- 0.04971818869570255 conf int (1.1858647055357527, 1.2853010829271565)

# test 12 sen (16, 8) 5 blocks L = 0
# Success rate: 0.8535 +- 0.015510482382981916 conf int:  (0.8379895176170181, 0.869010482382982)
# Average time:  7.773 +- 0.11167990118148374 conf int:  (7.661320098818516, 7.884679901181483)
# Average disturbance:  0.3681498569694612 +- 0.009016754622970968 conf int (0.3591331023464902, 0.3771666115924316)
# Success disturbance:  0.3236474210299748 +- 0.0032618483723223513 conf int (0.32038557265765244, 0.3269092694022969)

# re-run with better result and extra training
# Success rate: 0.98 +- 0.006140910465108007 conf int:  (0.973859089534892, 0.986140910465108)
# Average time:  5.823 +- 0.07360302837592858 conf int:  (5.749396971624072, 5.896603028375929)
# Average disturbance:  0.010949139570627202 +- 0.001631290638328977 conf int (0.009317848932298225, 0.012580430208956168)
# Success disturbance:  0.010208005482832187 +- 0.000629922803387627 conf int (0.00957808267944456, 0.010837928286219793)

# test 12 sen (16, 8) 5 blocks L = 1/16
# Success rate: 0.8675 +- 0.014871247807848764 conf int:  (0.8526287521921513, 0.8823712478078488)
# Average time:  16.534 +- 0.2593148276618358 conf int:  (16.274685172338163, 16.793314827661835)
# Average disturbance:  0.028398130496075098 +- 0.008862265467441038 conf int (0.01953586502863406, 0.03726039596351619)
# Success disturbance:  0.0048222664077216615 +- 0.0011388099962678462 conf int (0.0036834564114538153, 0.005961076403989516)

# Success rate: 0.854 +- 0.015488526136149261 conf int:  (0.8385114738638507, 0.8694885261361492)
# Average time:  16.55 +- 0.2681662044379607 conf int:  (16.28183379556204, 16.81816620443796)
# Average disturbance:  0.040633217357662545 +- 0.011175083417324427 conf int (0.029458133940338117, 0.05180830077498728)
# Success disturbance:  0.004905193826222548 +- 0.0009800345451694342 conf int (0.003925159281053114, 0.005885228371391984)

# test 12 sen (16, 8) 5 on 4 blocks L = 0
# Success rate: 0.919 +- 0.01196754295150404 conf int:  (0.907032457048496, 0.9309675429515041)
# Average time:  9.449 +- 0.26711267505688774 conf int:  (9.181887324943112, 9.716112675056888)
# Average disturbance:  0.0057402055636605704 +- 0.0003623202286170933 conf int (0.005377885335043477, 0.006102525792277688)
# Success disturbance:  0.005728231362187739 +- 0.0003781049873208481 conf int (0.005350126374866891, 0.006106336349508613)

# test 12 sen (16, 8) 5 on 4 blocks L = 1/16
# Success rate: 0.9065 +- 0.012770104152245887 conf int:  (0.8937298958477541, 0.9192701041522459)
# Average time:  15.7915 +- 0.1789079122453625 conf int:  (15.612592087754637, 15.970407912245362)
# Average disturbance:  0.004496943558502079 +- 0.0010990837149986782 conf int (0.0033978598435034013, 0.005596027273500751)
# Success disturbance:  0.00382379908777352 +- 0.00036445480525288277 conf int (0.003459344282520637, 0.004188253893026397)

# test 12 sen (16, 8) 5 on 6 blocks L = 0
# Success rate: 0.461 +- 0.021865004698006507 conf int:  (0.4391349953019935, 0.48286500469800653)
# Average time:  16.788 +- 0.630880708629924 conf int:  (16.157119291370076, 17.418880708629924)
# Average disturbance:  0.6121547830965992 +- 0.034980237807732406 conf int (0.5771745452888668, 0.6471350209043312)
# Success disturbance:  0.023061059049060688 +- 0.00491421059038254 conf int (0.018146848458678148, 0.02797526963944325)

# test 12 sen (16, 8) 5 on 6 blocks L = 1/16
# Success rate: 0.3785 +- 0.02127444494740449 conf int:  (0.3572255550525955, 0.3997744449474045)
# Average time:  25.643 +- 0.5299675200463554 conf int:  (25.113032479953645, 26.172967520046356)
# Average disturbance:  0.48704676733774965 +- 0.037302175494445755 conf int (0.4497445918433039, 0.5243489428321952)
# Success disturbance:  0.016289753003335374 +- 0.0028219442530998257 conf int (0.013467808750235549, 0.019111697256435186)

# test 12 sen (16, 8) 5 on 3 blocks L = 0
# Success rate: 0.5675 +- 0.0217310503747421 conf int:  (0.5457689496252579, 0.5892310503747421)
# Average time:  28.1225 +- 0.5161960910994452 conf int:  (27.606303908900554, 28.638696091099444)
# Average disturbance:  0.002119417083370595 +- 0.0008126642888596387 conf int (0.0013067527945109562, 0.0029320813722302215)
# Success disturbance:  0.002314757838702732 +- 0.000266661733792238 conf int (0.002048096104910494, 0.0025814195724949683)

# test 12 sen (16, 8) 5 on 3 blocks L = 1/16
# Success rate: 0.8545 +- 0.015466507620592318 conf int:  (0.8390334923794077, 0.8699665076205924)
# Average time:  21.416 +- 0.3413339866199472 conf int:  (21.074666013380053, 21.757333986619948)
# Average disturbance:  0.0028107170080664224 +- 0.0002515461714122707 conf int (0.0025591708366541517, 0.003062263179478682)
# Success disturbance:  0.0027929773389476476 +- 0.00027762764337714617 conf int (0.0025153496955705014, 0.003070604982324798)

# test 12 sen (16, 8) 5 on 2 blocks L = 0
# Success rate: 0.002 +- 0.0019596792682652357 conf int:  (4.03207317347643e-05, 0.003959679268265235)
# Average time:  39.922 +- 0.06883820380322447 conf int:  (39.85316179619677, 39.99083820380322)
# Average disturbance:  0.0 +- nan conf int (nan, nan)
# Success disturbance:  0.0 +- nan conf int (nan, nan)

# test 12 sen (16, 8) 5 on 2 blocks L = 1/16
# Success rate: 0.008 +- 0.003907559136338084 conf int:  (0.004092440863661916, 0.011907559136338085)
# Average time:  39.702 +- 0.13993944094669075 conf int:  (39.56206055905331, 39.84193944094669)
# Average disturbance:  0.0 +- nan conf int (nan, nan)
# Success disturbance:  0.0 +- nan conf int (nan, nan)

# test 13 sen (16, 8) 5 on 5 blocks L = 0
# Success rate: 0.9105 +- 0.012521496095775464 conf int:  (0.8979785039042245, 0.9230214960957754)
# Average time:  9.9375 +- 0.17268351374092283 conf int:  (9.764816486259077, 10.110183513740923)
# Average disturbance:  0.031745999754874236 +- 0.009253288699421998 conf int (0.02249271105545224, 0.04099928845429626)
# Success disturbance:  0.0114556862552186 +- 0.0015125912115646692 conf int (0.00994309504365393, 0.012968277466783213)

# test 13 sen (16, 8) 5 on 6 blocks L = 0
# Success rate: 0.895 +- 0.013446550799047885 conf int:  (0.8815534492009521, 0.9084465507990479)
# Average time:  9.5035 +- 0.21250561659866385 conf int:  (9.290994383401337, 9.716005616598665)
# Average disturbance:  0.054649128349937806 +- 0.013889102768612611 conf int (0.040760025581325195, 0.06853823111855069)
# Success disturbance:  0.011694700964264042 +- 0.001624865698473591 conf int (0.010069835265790451, 0.013319566662737675)

# test 13 sen (16, 8) 5 on 7 blocks L = 0
# Success rate: 0.846 +- 0.015832528251185973 conf int:  (0.830167471748814, 0.861832528251186)
# Average time:  10.2285 +- 0.3109381470774597 conf int:  (9.91756185292254, 10.53943814707746)
# Average disturbance:  0.13853160713870916 +- 0.024943271106540488 conf int (0.11358833603216867, 0.1634748782452497)
# Success disturbance:  0.009236035544582938 +- 0.0016193564239254691 conf int (0.007616679120657469, 0.010855391968508415)

# test 13 sen (16, 8) 5 on 8 blocks L = 0
# Success rate: 0.736 +- 0.019335063392890817 conf int:  (0.7166649366071092, 0.7553350633928908)
# Average time:  12.302 +- 0.4537731743021993 conf int:  (11.8482268256978, 12.755773174302199)
# Average disturbance:  0.3269883956632031 +- 0.03955075614415737 conf int (0.2874376395190457, 0.366539151807361)
# Success disturbance:  0.01052850470809569 +- 0.0032435850221807835 conf int (0.007284919685914907, 0.013772089730276464)

# test 13 sen (16, 8) 5 on 9 blocks L = 0
# Success rate: 0.594 +- 0.021540757371135588 conf int:  (0.5724592426288644, 0.6155407573711356)
# Average time:  16.512 +- 0.6077875339812202 conf int:  (15.90421246601878, 17.11978753398122)
# Average disturbance:  0.7493469187832684 +- 0.057979095631427 conf int (0.6913678231518414, 0.8073260144146921)
# Success disturbance:  0.012414259502533866 +- 0.0049254076406811806 conf int (0.007488851861852685, 0.017339667143215044)

# test 13 sen (16, 8) 5 on 4 blocks L = 0
# Success rate: 0.911 +- 0.012489898858073611 conf int:  (0.8985101011419264, 0.9234898988580736)
# Average time:  10.965 +- 0.15377805510254028 conf int:  (10.81122194489746, 11.11877805510254)
# Average disturbance:  0.013430563061099994 +- 0.004032643591494649 conf int (0.009397919469605345, 0.01746320665259463)
# Success disturbance:  0.00937026422465695 +- 0.0007132210064149554 conf int (0.008657043218241995, 0.010083485231071903)

# test 13 sen (16, 8) 5 on 3 blocks L = 0
# Success rate: 0.892 +- 0.013614416562135023 conf int:  (0.878385583437865, 0.905614416562135)
# Average time:  13.416 +- 0.24412766932182528 conf int:  (13.171872330678175, 13.660127669321826)
# Average disturbance:  0.008059876707594239 +- 0.0006583720506621307 conf int (0.007401504656932108, 0.008718248758256384)
# Success disturbance:  0.008605954868934068 +- 0.0007257823276632918 conf int (0.007880172541270776, 0.009331737196597373)

# test 13 sen (16, 8) 5 on 2 blocks L = 0
# Success rate: 0.7685 +- 0.018501296505900933 conf int:  (0.749998703494099, 0.7870012965059009)
# Average time:  20.8295 +- 0.44121776417124536 conf int:  (20.388282235828754, 21.270717764171245)
# Average disturbance:  0.006092503169449215 +- 0.0014717628119025744 conf int (0.004620740357546641, 0.007564265981351814)
# Success disturbance:  0.0064599417815783015 +- 0.0006213423883730173 conf int (0.005838599393205284, 0.007081284169951305)

# test 10 sen (16, 8) 4 on row t7 L = 0
# Success rate: 0.7675 +- 0.018529145874210062 conf int:  (0.7489708541257899, 0.78602914587421)
# Average time:  10.3915 +- 0.44867434724150534 conf int:  (9.942825652758495, 10.840174347241506)
# Average disturbance:  0.25617523169149164 +- 0.010960274602593856 conf int (0.2452149570888978, 0.26713550629408583)
# Success disturbance:  0.28002997425409876 +- 0.012982077176250828 conf int (0.26704789707784793, 0.2930120514303509)
