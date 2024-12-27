from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from nn4stud import L_BOUND, U_BOUND, DlNet, q

type vector = NDArray[np.floating]

MAX_WORKERS = 8

np.seterr(over="ignore")
np.seterr(under="ignore")
np.seterr(invalid="raise")


def format_float(num: float):
    return f"{num:.4f}".replace(".", ",")


def generate_plot(
    filename: str,
    x: vector,
    y: vector,
    predicted_y: vector,
    path: Path,
):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    plt.plot(x, y, "r")
    plt.plot(x, predicted_y, "b")

    plt.savefig(path / f"{filename}.png")
    plt.close()


def save_quality_file(
    param_arr: vector,
    loss_arr: vector,
    path: Path,
):
    with open(path / "quality.txt", "w") as file:
        for layer_size, loss_val in zip(param_arr, loss_arr):
            file.write(f"{layer_size}: {format_float(loss_val)}\n")


def create_train_network(
    x: vector, y: vector, layer_size: int, epochs: int, learning_rate: float, batch: int
) -> tuple[DlNet, float]:
    nn = DlNet(layer_size, learning_rate)
    train_x, train_y = generate_training_set(x, y, 2000)
    loss_val: float = nn.train(train_x, train_y, batch, epochs)
    return nn, loss_val


def generate_training_set(x: vector, y: vector, size: int) -> tuple[vector, vector]:
    train_indices = np.random.choice(len(x), 2000, replace=False)
    train_x = x[train_indices]
    train_y = y[train_indices]
    return train_x, train_y


def generate_ls_impact_data(
    x: vector,
    y: vector,
    layer_size_arr: vector,
    epochs: int,
    learning_rate: float,
    batch: int = 100,
):
    save_dir = Path(__file__).parent / f"data/layer_size_impact/{epochs}_epochs/"
    save_dir.mkdir(parents=True, exist_ok=True)

    predicted_y_arr: vector[vector] = np.array(
        [np.zeros_like(y) for _ in layer_size_arr]
    )
    loss_arr: vector = np.zeros(layer_size_arr.size)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                create_train_network, x, y, ls, epochs, learning_rate, batch
            )
            for ls in layer_size_arr
        ]

        for i, future in enumerate(futures):
            try:
                nn, loss_val = future.result()
                loss_arr[i] = loss_val
                predicted_y_arr[i] = nn.predict(x).flatten()
            except Exception as e:
                print(e)
                print(layer_size_arr[i], epochs, learning_rate, batch)
                continue

    for predicted_y, layer_size in zip(predicted_y_arr, layer_size_arr):
        generate_plot(
            f"{layer_size}_neurons",
            x,
            y,
            predicted_y,
            save_dir,
        )

    save_quality_file(layer_size_arr, loss_arr, save_dir)


def generate_lr_impact_data(
    x: vector, y: vector, lr_arr: vector, layer_size: int, epochs: int, batch: int = 100
):
    save_dir = Path(__file__).parent / f"data/learning_rate_impact/{epochs}_epochs/"
    save_dir.mkdir(parents=True, exist_ok=True)

    predicted_y_arr: vector[vector] = np.array([np.zeros_like(y) for _ in lr_arr])
    loss_arr: vector = np.zeros(lr_arr.size)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(create_train_network, x, y, layer_size, epochs, lr, batch)
            for lr in lr_arr
        ]

        for i, future in enumerate(futures):
            try:
                nn, loss_val = future.result()
                loss_arr[i] = loss_val
                predicted_y_arr[i] = nn.predict(x).flatten()
            except Exception as e:
                print(e)
                print(lr_arr[i], epochs, layer_size, batch)
                continue

    for predicted_y, lr in zip(predicted_y_arr, lr_arr):
        generate_plot(
            f"{lr}_lr",
            x,
            y,
            predicted_y,
            save_dir,
        )

    save_quality_file(lr_arr, loss_arr, save_dir)


def generate_epoch_impact_data(
    x: vector,
    y: vector,
    epochs_arr: vector,
    layer_size: int,
    learning_rate: float,
    batch: int = 100,
):
    save_dir = Path(__file__).parent / "data/epochs_impact/"
    save_dir.mkdir(parents=True, exist_ok=True)

    predicted_y_arr: vector[vector] = np.array([np.zeros_like(y) for _ in epochs_arr])
    loss_arr: vector = np.zeros(epochs_arr.size)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                create_train_network,
                x,
                y,
                layer_size,
                int(epochs),
                learning_rate,
                batch,
            )
            for epochs in epochs_arr
        ]

        for i, future in enumerate(futures):
            try:
                nn, loss_val = future.result()
                loss_arr[i] = loss_val
                predicted_y_arr[i] = nn.predict(x).flatten()
            except Exception as e:
                print(e)
                print(epochs_arr[i], learning_rate, layer_size, batch)
                continue

    for predicted_y, lr in zip(predicted_y_arr, epochs_arr):
        generate_plot(
            f"{lr}_lr",
            x,
            y,
            predicted_y,
            save_dir,
        )

    save_quality_file(epochs_arr, loss_arr, save_dir)


def learning_rate_impact():
    x: vector = np.linspace(L_BOUND, U_BOUND, 10000)
    y: vector = q(x)

    lr_arr = np.array([0.1, 0.15, 0.001, 0.0001])

    for epochs in [1000, 2500, 5000]:
        generate_lr_impact_data(x, y, lr_arr, 15, epochs)


def layer_size_impact():
    x: vector = np.linspace(L_BOUND, U_BOUND, 10000)
    y: vector = q(x)

    layer_size_arr: vector = np.array([1, 2, 3] + list(range(5, 76, 5)))

    for epochs in [1000, 2500, 5000]:
        generate_ls_impact_data(x, y, layer_size_arr, epochs, learning_rate=0.1)


def epochs_impact():
    x: vector = np.linspace(L_BOUND, U_BOUND, 10000)
    y: vector = q(x)

    epochs_arr = np.array([100, 200, 500, 1000, 2500, 5000])

    generate_epoch_impact_data(x, y, epochs_arr, 15, 0.1)


if __name__ == "__main__":
    layer_size_impact()
    learning_rate_impact()
    epochs_impact()
