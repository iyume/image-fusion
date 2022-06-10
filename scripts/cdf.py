from enum import Enum
from typing import Dict, List, Optional, Union, cast, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats


@overload
def load_excel_3d(filename: List[str]) -> np.ndarray:
    ...


@overload
def load_excel_3d(filename: str, sheet_names: Optional[List[str]] = None) -> np.ndarray:
    ...


def load_excel_3d(
    filename: Union[str, List[str]], sheet_names: Optional[List[str]] = None
) -> np.ndarray:
    """Loading data to 3-D numpy array.

    Expect input one excel with multi sheets, or multi excel with one sheet.

    In one excel input, sheets are different codes(C), rows are different indicators(I),
    columns are different test case(N), then the return shape is (C,I,N).

    Returns:
        np.ndarray: 3-D numpy array.
    """
    df_lst: List[pd.DataFrame]
    if isinstance(filename, list):
        df_lst = [pd.read_excel(i) for i in filename]
    else:
        excelfile = pd.ExcelFile(filename, engine="openpyxl")
        if sheet_names is None:
            sheet_names = excelfile.sheet_names
        df_lst = list(excelfile.parse(sheet_name=sheet_names).values())  # type: ignore
    assert all(i.shape == df_lst[0].shape for i in df_lst)
    return np.stack(df_lst)


class I(str, Enum):
    EN = "EN"
    MI = "MI"
    VIF = "VIF"
    SF = "SF"
    SD = "SD"
    QABF = "Qabf"
    SCD = "SCD"
    AG = "AG"


# plots order
indicators = (I.EN, I.MI, I.VIF, I.SF, I.SD, I.QABF, I.SCD, I.AG)


testset = "TNO"  # maybe mapping
avg_group = 20  # avg n elements
codes = (
    "densefuse",
    "FusionGAN",
    "GANMcC",
    "IFCNN",
    "PMGI",
    "rfn-nest",
    "U2Fusion",
    "ours",
)
data: np.ndarray = load_excel_3d([f"data/{i}_{testset}.xlsx" for i in codes])
# reindex
data = np.take(
    data,
    tuple(
        indicators.index(i)
        for i in ("EN", "SF", "SD", "MI", "VIF", "Qabf", "SCD", "AG")
    ),
    axis=1,
)


# data must be (C,I,N)
assert len(codes) == data.shape[0]
assert len(indicators) == data.shape[1]
data = data.transpose(1, 0, 2)  # (I,C,N)
data = np.sort(data)

print(f"data shape (I,C,N): {data.shape}")
if data.shape[2] != avg_group:
    assert data.shape[2] % avg_group == 0
    data = np.mean(data.reshape(data.shape[0], data.shape[1], avg_group, -1), axis=-1)
    print(f"data shape (avg to {avg_group}): {data.shape}")

data_dict: Dict[str, np.ndarray] = dict(zip((i.value for i in indicators), iter(data)))
labels = codes


def cdf(x: np.ndarray) -> np.ndarray:
    assert len(x.shape) == 1
    out = scipy.stats.norm.cdf(x, np.mean(x), np.std(x))
    return out


fig, axes = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(25, 15))
axes = cast("list[plt.Axes]", axes.flatten())
axes[-1].remove()
# fig.set_facecolor("white")
plt.subplots_adjust(wspace=0.5)

for i, title in enumerate(data_dict):
    axes[i].set_title(title)
    axes[i].set_xlabel("Cumulative Distributions")
    axes[i].set_ylabel("Values of The Metirc")
    axes[i].set_xlim(-0.05, 1.05)

line_style = ("-+", "-x", "-v", "-h", "-*", "-s", "-o", "-d")
if len(line_style) != len(data_dict):
    raise RuntimeError(
        f"filling the line style to suit {len(data_dict)} dataset first..."
    )

for i in range(len(labels)):
    for j, data in enumerate(data_dict.values()):
        y = data[i]
        x = cdf(y)
        axes[j].plot(
            x,
            y,
            line_style[i],
            markerfacecolor="none",
            label=f"{labels[i]} {np.mean(y):.2f}",
        )
        axes[j].legend(bbox_to_anchor=(1.04, 1))


fig.savefig(f"figure_{testset}.png", dpi=100, bbox_inches="tight")

...
