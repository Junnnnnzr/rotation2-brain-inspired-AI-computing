# 让BFS生成训练答案

from pathlib import Path
import torch

from navigation_core import (
    BLUE_TARGET_POSITION,
    GRID_SIZE,
    OBSTACLES,
    RED_TARGET_POSITION,
    find_shortest_path,
)

from snn_policy import (
    ACTION_TO_LABEL,
    LABEL_TO_ACTION,
    build_state,
)

# 枚举所有合法机器人位置，使用BFS生成正确的下一步动作
def generate_dataset():
    states = []
    labels = []
    sample_information = []

    targets = {
        "RED": RED_TARGET_POSITION,
        "BLUE": BLUE_TARGET_POSITION,
    }

    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            robot_position = (
                row,
                column
            )
            if robot_position in OBSTACLES:
                continue
            for target_name, target_position in (
                    targets.items()
            ):
                # 已经到达目标时不需要动作标签
                if robot_position == target_position:
                    continue
                shortest_path = find_shortest_path(
                    robot_position,
                    target_position
                )
                if not shortest_path:
                    continue
                # 不让神经网络一次预测完整路径，只让他预测当前状态的下一步
                correct_action = shortest_path[0]
                state = build_state(
                    robot_position,
                    target_position
                )
                label = ACTION_TO_LABEL[
                    correct_action
                ]
                states.append(state)
                labels.append(label)
                sample_information.append({
                    "robot_position": robot_position,
                    "target_name": target_name,
                    "target_position": target_position,
                    "correct_action": correct_action,
                })
    states = torch.stack(states)

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    return (
        states,
        labels,
        sample_information
    )

def main():
    print(
        "===== Generate BFS Supervised Dataset ====="
    )

    states, labels, sample_information = (
        generate_dataset()
    )

    print("Number of samples:", len(states))
    print("States shape:", states.shape)
    print("Labels shape:", labels.shape)

    print("\nAction label distribution:")

    for label, action in LABEL_TO_ACTION.items():
        count = (
            labels == label
        ).sum().item()

        print(
            f"{action}: {count}"
        )

    print("\nFirst five samples:")

    for index in range(
            min(5, len(sample_information))
    ):
        information = sample_information[
            index
        ]

        print(
            f"Sample {index}: "
            f"{information}"
        )

    data_directory = (
        Path(__file__).resolve().parent
        / "data"
    )

    data_directory.mkdir(
        exist_ok=True
    )

    output_path = (
        data_directory
        / "navigation_dataset.pt"
    )

    torch.save(
        {
            "states": states,
            "labels": labels,
        },
        output_path
    )

    print(
        "\nDataset saved to:",
        output_path
    )


if __name__ == "__main__":
    main()