import matplotlib.pyplot as plt
import numpy as np

def barh(names, values, title, xlabel, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    pos = np.arange(len(names))
    ax.barh(pos, values)
    ax.set_yticks(pos)
    ax.set_yticklabels(names)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def line(x, y, title, xlabel, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
