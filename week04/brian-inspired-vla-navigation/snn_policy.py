# 类脑计算
# 动作和标签的对应关系
# 环境状态构造方法
# 概率脉冲编码方法
# SNN网络结构

import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import surrogate

from navigation_core import (
    GRID_SIZE,
    OBSTACLES,
)


ACTION_TO_LABEL = {
    "UP": 0,
    "DOWN": 1,
    "LEFT": 2,
    "RIGHT": 3,
}

LABEL_TO_ACTION = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}

# 构造3×8×8的环境状态：机器人，目标，障碍物
def build_state(robot_position,target_position):
    state = torch.zeros(
        3,
        GRID_SIZE,
        GRID_SIZE,
        dtype=torch.float32
    )
    robot_row, robot_column = (
        robot_position
    )
    target_row, target_column = (
        target_position
    )
    state[
        0,
        robot_row,
        robot_column
    ] = 1.0
    state[
        1,
        target_row,
        target_column
    ] = 1.0
    for obstacle_row, obstacle_column in OBSTACLES:
        state[
            2,
            obstacle_row,
            obstacle_column
        ] = 1.0
    return state

# 将环境状态转换为概率脉冲序列
def encode_states_to_spikes(
        states,
        time_steps=16,
        active_probability=0.8,
        inactive_probability=0.02
):
    spike_probabilities = (
        states * active_probability
        + (1.0 - states)
        * inactive_probability
    )

    random_values = torch.rand(
        (
            time_steps,
            *states.shape
        ),
        device=states.device
    )

    input_spikes = (
        random_values
        < spike_probabilities.unsqueeze(0)
    ).float()

    return input_spikes

# 网络结构
class NavigationSNN(nn.Module):
    def __init__(
            self,
            hidden_size=64
    ):
        super().__init__()
        input_size = (
            3
            * GRID_SIZE
            * GRID_SIZE
        )
        output_size = 4

        self.input_layer = nn.Linear(
            input_size,
            hidden_size
        )

        self.hidden_neuron = neuron.LIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan(
                alpha=2.0
            ),
            detach_reset=True,
            step_mode="m"
        )

        self.output_layer = nn.Linear(
            hidden_size,
            output_size
        )

        self.output_neuron = neuron.LIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan(
                alpha=2.0
            ),
            detach_reset=True,
            step_mode="m"
        )

    def forward(self, input_sequence):
        flattened_sequence = (
            input_sequence.flatten(
                start_dim=2
            )
        )

        hidden_current = self.input_layer(
            flattened_sequence
        )

        hidden_spikes = self.hidden_neuron(
            hidden_current
        )

        output_current = self.output_layer(
            hidden_spikes
        )

        output_spikes = self.output_neuron(
            output_current
        )

        return output_spikes