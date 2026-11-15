# 保存公共导航功能：
# 地图配置
# 位置判断
# 机器人移动
# 语言解析
# BFS路径规划
# 环境绘制

import numpy as np
from collections import deque
import cv2

# 网格参数
# 地图：8×8个格子
# 每格：70×70像素
GRID_SIZE = 8
CELL_SIZE = 70
IMAGE_SIZE = GRID_SIZE * CELL_SIZE

WHITE = (255, 255, 255)
GRID_COLOR = (190, 190, 190)
OBSTACLE_COLOR = (100, 100, 100)
ROBOT_COLOR = (0, 0, 0)
RED_TARGET_COLOR = (0, 0, 255)
BLUE_TARGET_COLOR = (255, 0, 0)

# 初始位置和目标位置
INITIAL_ROBOT_POSITION = (6, 1)
RED_TARGET_POSITION = (0, 6)
BLUE_TARGET_POSITION = (6, 6)

# 固定障碍物
OBSTACLES = {
    (1, 3),
    (2, 3),
    (3, 3),
    (5, 3),
    (5, 4),
    (5, 5),
}

# 四个离散动作及其坐标变化
ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

# 将网格坐标转换为对应格子的中心像素坐标
def get_cell_center(position):
    row, column = position
    center_x = (
        column * CELL_SIZE
        + CELL_SIZE // 2
    )
    center_y = (
        row * CELL_SIZE
        + CELL_SIZE // 2
    )
    return center_x, center_y

# 判断位置是否位于地图内，并且不是障碍物
def is_valid_position(position):
    row, column = position
    inside_grid = (
        0 <= row < GRID_SIZE
        and 0 <= column < GRID_SIZE
    )
    if not inside_grid:
        return False
    if position in OBSTACLES:
        return False
    return True

# 根据动作计算机器人的新位置
def move_robot(robot_position, action):
    row_change, column_change = ACTIONS[action]
    current_row, current_column = robot_position
    new_position = (
        current_row + row_change,
        current_column + column_change,
    )
    if is_valid_position(new_position):
        return new_position, True
    return robot_position, False

# 从中文指令中识别目标颜色
def parse_language_command(command):
    normalized_command = (
        command.strip().replace(" ", "")
    )
    contains_red = "红" in normalized_command
    contains_blue = "蓝" in normalized_command
    if contains_red and contains_blue:
        raise ValueError(
            "指令中同时包含红色和蓝色，无法确定目标。"
        )
    if contains_red:
        return "RED"
    if contains_blue:
        return "BLUE"
    raise ValueError(
        "无法识别目标颜色，请在指令中包含“红色”或“蓝色”。"
    )

# 根据目标名称返回目标的网格坐标
def get_target_position(target_name):
    if target_name == "RED":
        return RED_TARGET_POSITION
    if target_name == "BLUE":
        return BLUE_TARGET_POSITION
    raise ValueError(
        f"Unknown target name: {target_name}"
    )

# 使用BFS寻找从起点到目标的最短动作序列
def find_shortest_path(
        start_position,
        target_position
):
    search_queue = deque()
    search_queue.append(
        (start_position, [])
    )
    visited_positions = {
        start_position
    }

    action_order = [
        "UP",
        "DOWN",
        "LEFT",
        "RIGHT",
    ]

    while search_queue:
        current_position, current_path = (
            search_queue.popleft()
        )
        if current_position == target_position:
            return current_path
        current_row, current_column = (
            current_position
        )
        for action in action_order:
            row_change, column_change = (
                ACTIONS[action]
            )
            next_position = (
                current_row + row_change,
                current_column + column_change,
            )
            if not is_valid_position(next_position):
                continue
            if next_position in visited_positions:
                continue
            visited_positions.add(next_position)
            next_path = (
                current_path + [action]
            )
            search_queue.append(
                (next_position, next_path)
            )
    return None

# 根据机器人当前位置绘制完整的栅格环境
def render_environment(robot_position):
    image = np.full(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ),
        fill_value=255,
        dtype=np.uint8,
    )
    # 绘制障碍物
    for row, column in OBSTACLES:
        x1 = column * CELL_SIZE + 5
        y1 = row * CELL_SIZE + 5
        x2 = (
            (column + 1) * CELL_SIZE
            - 5
        )
        y2 = (
            (row + 1) * CELL_SIZE
            - 5
        )
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            OBSTACLE_COLOR,
            thickness=-1,
        )
    # 绘制红色目标
    red_center = get_cell_center(
        RED_TARGET_POSITION
    )
    cv2.circle(
        image,
        red_center,
        CELL_SIZE // 4,
        RED_TARGET_COLOR,
        thickness=-1,
    )
    # 绘制蓝色目标
    blue_center = get_cell_center(
        BLUE_TARGET_POSITION
    )
    cv2.circle(
        image,
        blue_center,
        CELL_SIZE // 4,
        BLUE_TARGET_COLOR,
        thickness=-1,
    )
    # 绘制机器人
    robot_center = get_cell_center(
        robot_position
    )
    cv2.circle(
        image,
        robot_center,
        CELL_SIZE // 4,
        ROBOT_COLOR,
        thickness=-1,
    )
    # 绘制网格线
    for index in range(GRID_SIZE + 1):
        coordinate = min(
            index * CELL_SIZE,
            IMAGE_SIZE - 1
        )
        cv2.line(
            image,
            (0, coordinate),
            (
                IMAGE_SIZE - 1,
                coordinate
            ),
            GRID_COLOR,
            thickness=1,
        )
        cv2.line(
            image,
            (coordinate, 0),
            (
                coordinate,
                IMAGE_SIZE - 1
            ),
            GRID_COLOR,
            thickness=1,
        )
    return image