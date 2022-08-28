# -*- coding:utf-8  -*-
# Time  : 2022/8/28 下午20:23
# Author: observer

"""
# =================================== Important =========================================
Notes:
1. this agents is random agents , which can fit any env in Jidi platform.
2. if you want to load .pth file, please follow the instruction here:
https://github.com/jidiai/ai_lib/blob/master/examples/demo
"""
import os
import pickle
import numpy as np
import torch

tar = -2.0
def my_controller(observation, action_space, is_act_continuous=True):
    agent_action = []
    state = observation['obs']
    gap = state - tar
    if gap > 10.0:
        gap = 10.0
    if gap < 0:
        gap = 0
    for i in range(len(action_space)):
        action_ = [gap]
        agent_action.append(np.float32(action_))
        #print(action_)
    print(agent_action)
    return agent_action

"""
def sample_single_dim(action_space_list_each, is_act_continuous):
    each = []
    if is_act_continuous:
        each = action_space_list_each.sample()
    else:
        if action_space_list_each.__class__.__name__ == "Discrete":
            each = [0] * action_space_list_each.n
            idx = action_space_list_each.sample()
            each[idx] = 1
        elif action_space_list_each.__class__.__name__ == "MultiDiscreteParticle":
            each = []
            nvec = action_space_list_each.high - action_space_list_each.low + 1
            sample_indexes = action_space_list_each.sample()

            for i in range(len(nvec)):
                dim = nvec[i]
                new_action = [0] * dim
                index = sample_indexes[i]
                new_action[index] = 1
                each.extend(new_action)
        elif action_space_list_each.__class__.__name__ == "Discrete_SC2":
            each = action_space_list_each.sample()
        elif action_space_list_each.__class__.__name__ == "Box":
            each = action_space_list_each.sample()
    return each
"""