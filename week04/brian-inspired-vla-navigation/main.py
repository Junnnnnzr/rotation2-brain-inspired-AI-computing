# 项目最终运行入口

from pathlib import Path
import cv2
import torch
from spikingjelly.activation_based import (
    functional as sj_functional
)

from navigation_core import (
    INITIAL_ROBOT_POSITION,
    get_target_position,
    move_robot,
    parse_language_command,
    render_environment,
)

from snn_policy import (
    LABEL_TO_ACTION,
    NavigationSNN,
    build_state,
    encode_states_to_spikes,
)

# 创建SNN并加载训练好的最佳参数
def load_trained_model(device):
    model_path = (
        Path(__file__).resolve().parent
        / "models"
        / "navigation_snn.pth"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    hidden_size = checkpoint[
        "hidden_size"
    ]

    time_steps = checkpoint[
        "time_steps"
    ]

    model = NavigationSNN(
        hidden_size=hidden_size
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    saved_accuracy = checkpoint[
        "test_accuracy"
    ]

    print(
        "Loaded model test accuracy:",
        f"{saved_accuracy * 100:.2f}%"
    )

    return model, time_steps

# 使用训练后的SNN预测下一步动作
def predict_action(
        model,
        robot_position,
        target_position,
        device,
        time_steps,
        inference_repeats=5
):
    state = build_state(
        robot_position,
        target_position
    )

    state = (
        state
        .unsqueeze(0)
        .to(device)
    )

    accumulated_rates = torch.zeros(
        4,
        device=device
    )

    with torch.no_grad():
        for _ in range(inference_repeats): # 每次输入脉冲是随机的，对同一状态执行5次预测求平均，以减少单次随机的偶然错误
            input_spikes = (
                encode_states_to_spikes(
                    state,
                    time_steps=time_steps,
                    active_probability=0.8,
                    inactive_probability=0.02
                )
            )

            output_spikes = model(
                input_spikes
            )

            output_rates = (
                output_spikes.mean(dim=0)
            )

            accumulated_rates += (
                output_rates[0]
            )

            sj_functional.reset_net(
                model
            )

    average_rates = (
        accumulated_rates
        / inference_repeats
    )

    selected_label = int(
        average_rates.argmax().item()
    )

    selected_action = LABEL_TO_ACTION[
        selected_label
    ]

    return (
        selected_action,
        average_rates.cpu()
    )

def main():
    print(
        "===== Trained SNN Navigation ====="
    )

    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    model, time_steps = load_trained_model(device)

    command = input("请输入机器人指令：")

    try:
        target_name = parse_language_command(command)

    except ValueError as error:
        print(f"指令解析失败：{error}")
        return

    target_position = get_target_position(target_name)

    robot_position = (INITIAL_ROBOT_POSITION)

    max_steps = 50
    navigation_succeeded = False

    print("Target:", target_name)
    print("Target position:", target_position)

    for step in range(max_steps):
        image = render_environment(robot_position)
        cv2.imshow("Trained SNN Navigation",image)

        cv2.waitKey(300)

        if robot_position == target_position:
            navigation_succeeded = True
            print( f"\n导航成功，共执行 {step} 步。")

            break

        selected_action, output_rates = (
            predict_action(
                model=model,
                robot_position=robot_position,
                target_position=target_position,
                device=device,
                time_steps=time_steps,
                inference_repeats=5
            )
        )
        old_position = robot_position

        robot_position, move_succeeded = (
            move_robot(robot_position,selected_action)
        )

        print(f"\nStep {step + 1}")

        print("Current position:",old_position)

        print("SNN output rates:",output_rates.tolist())

        print("SNN selected:",selected_action)

        print("Move succeeded:",move_succeeded)

        print("New position:",robot_position)

    else:
        print("\n达到最大步数，导航失败。")

    final_image = render_environment( robot_position)

    output_directory = (Path(__file__).resolve().parent/ "outputs")

    output_directory.mkdir(exist_ok=True)

    result_name = (
        "success"
        if navigation_succeeded
        else "failed")

    output_path = (
        output_directory
        / (
            f"trained_snn_"
            f"{target_name.lower()}_"
            f"{result_name}.png"
        )
    )

    cv2.imwrite(
        str(output_path),
        final_image
    )

    print(
        "\nFinal result saved to:",
        output_path
    )

    cv2.imshow(
        "Trained SNN Navigation",
        final_image
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()