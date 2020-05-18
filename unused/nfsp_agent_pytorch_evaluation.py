# Copyright 2019 Matthew Judell. All rights reserved.
# Copyright 2019 DATA Lab at Texas A&M University. All rights reserved.
# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

''' Neural Fictitious Self-Play (NFSP) agent implemented in TensorFlow.

See the paper https://arxiv.org/abs/1603.01121 for more details.
'''

import collections
import enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlcard.agents.dqn_agent_pytorch import DQNAgent
from rlcard.agents.nfsp_agent import ReservoirBuffer
from rlcard.agents.nfsp_agent_pytorch import NFSPAgent
from rlcard.utils.utils import remove_illegal
Transition = collections.namedtuple('Transition', 'info_state action_probs')

MODE = enum.Enum('mode', 'best_response average_policy')

class NFSPEvalAgent(NFSPAgent):


    def _act(self, info_state):
        ''' Predict action probability givin the observation and legal actions
            Not connected to computation graph
        Args:
            info_state (numpy.array): An obervation.

        Returns:
            action_probs (numpy.array): The predicted action probability.
        '''

        info_state = np.expand_dims(info_state, axis=0)
        info_state = torch.from_numpy(info_state).float().to(self.device)

        with torch.no_grad():
            log_action_probs = self.policy_network(info_state).numpy()

        action_probs = np.exp(log_action_probs)[0]
        print(info_state)
        print(action_probs)
        return action_probs
