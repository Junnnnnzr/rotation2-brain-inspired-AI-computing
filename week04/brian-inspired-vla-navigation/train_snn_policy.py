# 完成数据读取、训练、测试和模型保存
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from spikingjelly.activation_based import (
    functional as sj_functional
)
import torch.nn.functional as F

from snn_policy import (
    NavigationSNN,
    encode_states_to_spikes,
)

from torch.utils.data import (
    WeightedRandomSampler
)

# 从文件中读取环境状态和BFS动作标签
def load_navigation_dataset():
    data_path = (
        Path(__file__).resolve().parent
        / "data"
        / "navigation_dataset.pt"
    )

    dataset_file = torch.load(
        data_path,
        weights_only=True
    )

    states = dataset_file["states"]
    labels = dataset_file["labels"]

    dataset = TensorDataset(
        states,
        labels
    )

    return dataset

# 第一次训练测试得到结果：机器人只会up和down，因此需要让训练程序更频繁抽取left和right样本
# def create_data_loaders(
#         dataset,
#         batch_size=16
# ):
#     """
#     将数据集按照80%和20%划分，
#     并创建分批读取器。
#     """
#     train_size = int(
#         len(dataset) * 0.8
#     )
#
#     test_size = (
#         len(dataset) - train_size
#     )
#
#     random_generator = (
#         torch.Generator()
#         .manual_seed(0)
#     )
#
#     train_dataset, test_dataset = (
#         random_split(
#             dataset,
#             [train_size, test_size],
#             generator=random_generator
#         )
#     )
#
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True
#     )
#
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False
#     )
#
#     return (
#         train_loader,
#         test_loader,
#         train_dataset,
#         test_dataset
#     )

# 划分训练集和测试集，并使用加权采样平衡训练动作类别
def create_data_loaders(
        dataset,
        batch_size=16
):
    train_size = int(
        len(dataset) * 0.8
    )

    test_size = (
        len(dataset) - train_size
    )

    random_generator = (
        torch.Generator()
        .manual_seed(0)
    )

    train_dataset, test_dataset = (
        random_split(
            dataset,
            [train_size, test_size],
            generator=random_generator
        )
    )

    all_labels = dataset.tensors[1]

    train_indices = train_dataset.indices

    train_labels = all_labels[
        train_indices
    ]

    class_counts = torch.bincount(
        train_labels,
        minlength=4
    ).float()

    class_weights = (
        1.0
        / class_counts.clamp_min(1.0)
    )

    sample_weights = class_weights[
        train_labels
    ]

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    # 加权采样器：样本数量少，采样权重更大

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print(
        "Original training class counts:",
        class_counts.tolist()
    )

    print(
        "Training class weights:",
        class_weights.tolist()
    )

    return (
        train_loader,
        test_loader,
        train_dataset,
        test_dataset
    )

# 使用整个训练集训练一轮
def train_one_epoch(
        model,
        train_loader,
        optimizer,
        device,
        time_steps=16
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for states, labels in train_loader:
        states = states.to(device)
        labels = labels.to(device)

        input_spikes = encode_states_to_spikes(
            states,
            time_steps=time_steps,
            active_probability=0.8,
            inactive_probability=0.02
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output_spikes = model(
            input_spikes
        )

        output_rates = output_spikes.mean(
            dim=0
        )

        target_rates = F.one_hot(
            labels,
            num_classes=4
        ).float()

        loss = F.mse_loss(
            output_rates,
            target_rates
        )

        loss.backward()

        optimizer.step()

        predictions = output_rates.argmax(
            dim=1
        )

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

        sj_functional.reset_net(
            model
        )

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        total_correct / total_samples
    )

    return average_loss, accuracy

# 在测试集上评估网络
def evaluate(
        model,
        test_loader,
        device,
        time_steps=16
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for states, labels in test_loader:
            states = states.to(device)
            labels = labels.to(device)

            input_spikes = (
                encode_states_to_spikes(
                    states,
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

            target_rates = F.one_hot(
                labels,
                num_classes=4
            ).float()

            loss = F.mse_loss(
                output_rates,
                target_rates
            )

            predictions = output_rates.argmax(
                dim=1
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

            sj_functional.reset_net(
                model
            )

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        total_correct / total_samples
    )

    return average_loss, accuracy

def main():
    print(
        "===== Prepare Navigation Dataset ====="
    )

    dataset = load_navigation_dataset()

    (
        train_loader,
        test_loader,
        train_dataset,
        test_dataset
    ) = create_data_loaders(dataset)

    print("Total samples:", len(dataset))
    print(
        "Training samples:",
        len(train_dataset)
    )
    print(
        "Testing samples:",
        len(test_dataset)
    )

    batch_states, batch_labels = next(
        iter(train_loader)
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    hidden_size = 64
    time_steps = 16
    epochs = 100

    model = NavigationSNN(
        hidden_size=hidden_size
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    batch_states = batch_states.to(
        device
    )

    time_steps = 16

    input_sequence = encode_states_to_spikes(
        batch_states,
        time_steps=time_steps,
        active_probability=0.8,
        inactive_probability=0.02
    )

    output_spikes = model(
        input_sequence
    )

    output_rates = output_spikes.mean(
        dim=0
    )

    print(
        "Input sequence shape:",
        input_sequence.shape
    )

    print(
        "Output spikes shape:",
        output_spikes.shape
    )

    print(
        "Output rates shape:",
        output_rates.shape
    )

    print(
        "First sample output rates:",
        output_rates[0]
    )

    sj_functional.reset_net(model)

    model_directory = (
            Path(__file__).resolve().parent
            / "models"
    )

    model_directory.mkdir(
        exist_ok=True
    )

    model_path = (
            model_directory
            / "navigation_snn.pth"
    )

    best_test_accuracy = -1.0

    print("\n===== Start Training =====")

    for epoch in range(
            1,
            epochs + 1
    ):
        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                time_steps=time_steps
            )
        )

        test_loss, test_accuracy = evaluate(
            model=model,
            test_loader=test_loader,
            device=device,
            time_steps=time_steps
        )

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = (
                test_accuracy
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "hidden_size":
                        hidden_size,

                    "time_steps":
                        time_steps,

                    "test_accuracy":
                        best_test_accuracy,
                },
                model_path
            )

        if (
                epoch == 1
                or epoch % 10 == 0
        ):
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Train accuracy: "
                f"{train_accuracy * 100:.2f}% | "
                f"Test loss: {test_loss:.4f} | "
                f"Test accuracy: "
                f"{test_accuracy * 100:.2f}%"
            )

    print("\nTraining finished.")

    print(
        "Best test accuracy:",
        f"{best_test_accuracy * 100:.2f}%"
    )

    print(
        "Best model saved to:",
        model_path
    )

if __name__ == "__main__":
    main()