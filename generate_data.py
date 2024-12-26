from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from nn4stud import L_BOUND, U_BOUND, DlNet, q

type vector = NDArray[np.floating]

MAX_WORKERS = 8


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
    ls_arr: vector,
    test_loss_arr: vector,
    res_loss_arr,
    iters: int,
    lr: float,
    path: Path,
):
    with open(path / f"quality_{iters}_iters_{lr}_lr.txt", "w") as file:
        for layer_size, train_loss_val, test_loss_val in zip(
            ls_arr, test_loss_arr, res_loss_arr
        ):
            file.write(
                f"{layer_size}: training loss_val {format_float(train_loss_val)} - testing loss_val {format_float(test_loss_val)}\n"
            )


def create_train_network(
    x: vector, y: vector, layer_size: int, iters: int, lr: float
) -> tuple[DlNet, float]:
    nn = DlNet(layer_size, lr)
    loss_val: float = nn.train(x, y, iters)
    return nn, loss_val


def generate_data(
    train_x: vector,
    train_y: vector,
    test_x: vector,
    test_y: vector,
    layer_size_arr: vector,
    iters: int,
    lr: float,
):
    save_dir = Path(__file__).parent / f"data/{lr}_lr/{iters}_iters"
    save_dir.mkdir(parents=True, exist_ok=True)

    predicted_y_arr: vector[vector] = np.array(
        [np.zeros_like(test_y) for _ in layer_size_arr]
    )
    train_loss_arr: vector = np.zeros(layer_size_arr.size)
    test_loss_arr: vector = np.zeros(layer_size_arr.size)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(create_train_network, train_x, train_y, ls, iters, lr)
            for ls in layer_size_arr
        ]

        for i, future in enumerate(futures):
            nn, loss_val = future.result()
            train_loss_arr[i] = loss_val
            predicted_y_arr[i] = nn.predict(test_x).flatten()
            test_loss_arr[i] = nn.calc_total_loss_val(test_y, predicted_y_arr[i])

    for predicted_y, layer_size in zip(
        predicted_y_arr,
        layer_size_arr,
    ):
        generate_plot(
            f"{layer_size}_ls_{iters}_iters_{lr}_lr",
            test_x,
            test_y,
            predicted_y,
            save_dir,
        )

    save_quality_file(
        layer_size_arr, train_loss_arr, test_loss_arr, iters, lr, save_dir
    )


if __name__ == "__main__":
    train_x: vector = np.linspace(L_BOUND + 1, U_BOUND - 1, 100)
    train_y: vector = q(train_x)

    test_x: vector = np.linspace(L_BOUND, U_BOUND, 100)
    test_y: vector = q(test_x)

    iters_arr = [15000, 150000, 500000, 1000000]
    lr_arr = [0.003, 0.004, 0.005, 0.006]
    layer_size_arr: vector = np.array(list(range(5, 301, 5)))

    for lr in lr_arr:
        for iters in iters_arr:
            generate_data(train_x, train_y, test_x, test_y, layer_size_arr, iters, lr)
