import pathlib
from typing import Iterable, Optional
import numpy as np
import matplotlib.pyplot as plt

def save_plot(x: np.ndarray, y: Iterable[np.ndarray], save_file: pathlib.Path, x_label: Optional[str] = None, 
    y_label: Optional[str] = None, title: Optional[str] = None,
    legend: Optional[Iterable[str]] = None
    ) -> None:
    """
    Creates and saves plot of training metrics.
    
    :param: x: x-axis data
    :dtype: np.ndarray
    :param: y: y-axis data (metrics)
    :dtype: Iterable[np.ndarray]
    :param: save_file: path to save figure
    :dtype: pathlib.Path
    :param: x_label: optional x-axis label
    :dtype: Optional[str]
    :param: y_label: optional y-axis label
    :dtype: Optional[str]
    :param: title: optional title
    :dtype: Optional[str]
    :param: legend: optional legend
    :dtype: Optional[Iterable[str]]
    :return: None
    :rtype: None
    """
    fig, ax = plt.subplots()
    for y_tgt in y:
        ax.plot(x, y_tgt)
    if title:
        ax.title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if len(legend):
        ax.legend(legend)
    plt.savefig(save_file)