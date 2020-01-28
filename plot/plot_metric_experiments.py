# Copyright 2019 Julian Niedermeier & Goncalo Mordido
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
from collections import OrderedDict
import pathlib
import os
import sys

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
import numpy as np
import pandas as pd

sns.set(context="paper", style="white")
np.set_printoptions(suppress=True)

COLOR_PURPLE = "#800080"


def load_original(root, metric, network, dataset, tag="", swapped=False):
    tag = tag if tag else ""
    filename = f"{dataset}{tag}.swapped.npz" if swapped else f"{dataset}{tag}.npz"
    path = os.path.join(args.root, metric, metric_to_network[metric], dataset, filename)
    return np.load(path)


def load_data(root, metric, network, dataset, test_type, tag="", swapped=False):
    tag = tag if tag else ""
    test_type = f"swapped.{test_type}" if swapped else test_type
    search_path = os.path.join(root, metric, network, dataset, test_type)
    file_paths = sorted(
        [
            p
            for p in pathlib.Path(search_path).iterdir()
            if p.is_file() and p.suffix == ".npz" and tag in p.name
        ]
    )
    return [np.load(file) for file in file_paths]


def load_images(
    root,
    noise_type,
    noise_amounts,
    dataset,
    test_or_train="test",
    imagedir="distorted_images",
    dataset_entry_id=0,
    load_original=True,
):
    root = os.path.join(root, imagedir)

    images = {}
    if load_original:
        file = os.path.join(root, f"{dataset}_{test_or_train}_{dataset_entry_id}.png")
        images[0.0] = Image.open(file)

    for noise_amount in noise_amounts:
        for i in range(5):
            # File names might have trailing zeros for noise amount, scan for it here
            file = os.path.join(
                root,
                f"{dataset}_{test_or_train}_{dataset_entry_id}_{noise_type}-{noise_amount}{'0' * i}.png",
            )
            if os.path.exists(file):
                images[noise_amount] = Image.open(file)
                break
        if noise_amount not in images:
            raise ValueError(
                f"Image for {noise_type} with {noise_amount} at {file} does not exist!"
            )
    return images


def noise(
    host,
    fig,
    image_offset_id,
    root,
    metric_data,
    dataset,
    title,
    xlabel,
    noise_amounts,
    fid_ylim,
    images,
    **kwargs,
):
    par1 = host.twinx()

    host.set_xlabel(xlabel)
    par1.set_ylabel("FID", rotation="horizontal")

    host.yaxis.label.set_color("k")
    par1.yaxis.label.set_color(COLOR_PURPLE)

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis="y", colors="k", **tkw)
    par1.tick_params(axis="y", colors=COLOR_PURPLE, **tkw)
    host.tick_params(axis="x", **tkw)

    noise_amounts = [0.0] + noise_amounts
    x = np.arange(len(noise_amounts))

    # fti
    precision_fti = np.array(
        [metric_data["fti"][noise_amount]["impact"] for noise_amount in noise_amounts]
    )
    precision_fti /= precision_fti[0]
    recall_fti = np.array(
        [metric_data["fti.swapped"][noise_amount]["impact"] for noise_amount in noise_amounts]
    )
    recall_fti /= recall_fti[0]

    # prd
    precision_prd = np.array(
        [
            metric_data["prd"][noise_amount]["f_beta_data"][1]
            for noise_amount in noise_amounts
        ]
    )
    precision_prd /= precision_prd[0]
    recall_prd = np.array(
        [
            metric_data["prd"][noise_amount]["f_beta_data"][0]
            for noise_amount in noise_amounts
        ]
    )
    recall_prd /= recall_prd[0]

    # impar
    precision_impar = np.array(
        [
            metric_data["impar"][noise_amount]["precision"]
            for noise_amount in noise_amounts
        ]
    )
    precision_impar /= precision_impar[0]
    recall_impar = np.array(
        [metric_data["impar"][noise_amount]["recall"] for noise_amount in noise_amounts]
    )
    recall_impar /= recall_impar[0]

    # is
    inception_score = np.array(
        [
            metric_data["inception_score"][noise_amount]["inception_score"][0]
            for noise_amount in noise_amounts
        ]
    )
    inception_score /= inception_score[0]

    # fid
    fid = np.array(
        [metric_data["fid"][noise_amount]["fid"] for noise_amount in noise_amounts]
    )
    fid /= fid[0]

    host.plot([], [], "k-", label="quality")
    host.plot([], [], "k--", label="diversity")
    host.plot([], [], ":", label="FID", color=COLOR_PURPLE)
    host.plot([], [], "c-.", label="IS")
    host.scatter([], [], marker="^", color="r", label="FTI")
    host.scatter([], [], marker="s", color="g", label="PRD")
    host.plot(x, precision_prd, "gs-")
    host.plot(x, recall_prd, "gs--")
    host.scatter([], [], marker="o", color="b", label="IMPAR")
    host.plot(x, precision_impar, "bo-")
    host.plot(x, recall_impar, "bo--")

    par1.plot(x, fid, ":", color=COLOR_PURPLE)
    host.plot(x, inception_score, "c-.")
    host.plot(x, precision_fti, "r^-")
    host.plot(x, recall_fti, "r^--")

    host.set_xticks(np.arange(len(x)))
    host.set_xticklabels(noise_amounts)
    par1.set_xticks(np.arange(len(x)))
    par1.set_xticklabels(noise_amounts)

    if image_offset_id in [0, 1]:
        yoffset = 640
    else:
        yoffset = 272
    if image_offset_id in [0, 2]:
        xoffset = 46
    else:
        xoffset = 539
    img_size = 77
    for i, noise_amount in enumerate(noise_amounts):
        xshift = 6
        image = images[noise_amount]
        image = image.resize((img_size, img_size))
        fig.figimage(
            image,
            xoffset + img_size * i + xshift * i,
            yoffset,
            zorder=10,
            cmap=plt.get_cmap("gray"),
            origin="upper",
        )

    host.set_ylim(0, 1.4)
    par1.set_ylim(*fid_ylim)

    xoffset = 0.05
    yoffset = 1.06
    par1.yaxis.set_label_coords(1.0 + xoffset, yoffset)

    [tick.set_visible(False) for tick in host.yaxis.get_major_ticks()[-2:]]

    host.set_title(title)
    host.legend(loc=3)


def plot_mode_dropping(
    host,
    root,
    metric_data,
    dataset,
    title,
    xlabel,
    filename,
    fid_ylim,
    dpi=600,
    legend_loc=None,
    **kwargs,
):
    has_host = True if host else False
    if not has_host:
        fig, host = plt.subplots()
        fig.subplots_adjust(right=0.76, left=0.08)

    par1 = host.twinx()

    host.set_xlabel(xlabel)
    par1.set_ylabel("FID", rotation="horizontal")

    host.yaxis.label.set_color("k")
    par1.yaxis.label.set_color(COLOR_PURPLE)

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis="y", colors="k", **tkw)
    par1.tick_params(axis="y", colors=COLOR_PURPLE, **tkw)
    host.tick_params(axis="x", **tkw)

    test_labels = list(metric_data["fid"].keys())
    if dataset == "cifar100":
        test_labels = test_labels[::10]
    # Could be used to plot range of used train labels
    # train_labels = metric_data["fid"][test_labels[0]]["allowed_train_labels"]
    # train_label_range = (train_labels.min(), train_labels.max())

    xticks = np.arange(len(test_labels))
    xlabels = xticks

    # fti
    precision_fti = np.array([metric_data["fti"][key]["impact"] for key in test_labels])
    precision_fti /= precision_fti[0]
    recall_fti = np.array(
        [metric_data["fti.swapped"][key]["impact"] for key in test_labels]
    )
    recall_fti /= recall_fti[0]

    # prd
    precision_prd = np.array(
        [metric_data["prd"][key]["f_beta_data"][1] for key in test_labels]
    )
    precision_prd /= precision_prd[0]
    recall_prd = np.array(
        [metric_data["prd"][key]["f_beta_data"][0] for key in test_labels]
    )
    recall_prd /= recall_prd[0]

    # impar
    precision_impar = np.array(
        [metric_data["impar"][key]["precision"] for key in test_labels]
    )
    precision_impar /= precision_impar[0]
    recall_impar = np.array(
        [metric_data["impar"][key]["recall"] for key in test_labels]
    )
    recall_impar /= recall_impar[0]

    # fid
    fid = np.array([metric_data["fid"][key]["fid"] for key in test_labels])
    fid /= fid[0]

    host.plot([], [], "k-", label="quality")
    host.plot([], [], "k--", label="diversity")
    host.plot([], [], ":", label="FID", color=COLOR_PURPLE)
    host.scatter([], [], marker="^", color="r", label="FTI")
    host.scatter([], [], marker="s", color="g", label="PRD")
    host.plot(xticks, precision_prd, "gs-")
    host.plot(xticks, recall_prd, "gs--")
    host.scatter([], [], marker="o", color="b", label="IMPAR")
    host.plot(xticks, precision_impar, "bo-")
    host.plot(xticks, recall_impar, "bo--")

    par1.plot(xticks, fid, ":", color=COLOR_PURPLE)
    host.plot(xticks, precision_fti, "r^-")
    host.plot(xticks, recall_fti, "r^--")

    host.set_xticks(xticks)
    host.set_xticklabels(xlabels)
    par1.set_xticks(xticks)
    par1.set_xticklabels(xlabels)

    host.set_ylim(0, 1.0)
    par1.set_ylim(*fid_ylim)

    xoffset = 0.05
    par1.yaxis.set_label_coords(1.0 + xoffset, 1.07)

    host.set_title(title)
    host.legend(loc=3)

    if not has_host:
        file = os.path.join(root, filename)
        plt.tight_layout()
        plt.savefig(file, dpi=dpi)
        plt.close()

    if legend_loc is not None:
        host.legend(loc=legend_loc)


def plot_mode_inventing(
    host,
    root,
    metric_data,
    dataset,
    title,
    xlabel,
    filename,
    fid_ylim,
    dpi=600,
    **kwargs,
):
    has_host = True if host else False
    if not has_host:
        fig, host = plt.subplots()
        fig.subplots_adjust(right=0.76, left=0.08)

    par1 = host.twinx()

    host.set_xlabel(xlabel)
    par1.set_ylabel("FID", rotation="horizontal")

    host.yaxis.label.set_color("k")
    par1.yaxis.label.set_color(COLOR_PURPLE)

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis="y", colors="k", **tkw)
    par1.tick_params(axis="y", colors=COLOR_PURPLE, **tkw)
    host.tick_params(axis="x", **tkw)

    test_labels = list(metric_data["fid"].keys())
    if dataset == "cifar100":
        test_labels = test_labels[::10][1:] + [test_labels[-1]]
    # Could be used to plot range of used train labels
    # train_labels = metric_data["fid"][test_labels[0]]["allowed_train_labels"]
    # train_label_range = (train_labels.min(), train_labels.max())

    xticks = np.arange(len(test_labels))
    xlabels = [f"[{key[0]}-{key[1]}]" for key in test_labels]

    normalization_id = len(test_labels) // 2 - 1
    # fti
    precision_fti = np.array([metric_data["fti"][key]["impact"] for key in test_labels])
    precision_fti /= precision_fti[normalization_id]
    recall_fti = np.array(
        [metric_data["fti.swapped"][key]["impact"] for key in test_labels]
    )
    recall_fti /= recall_fti[normalization_id]

    # prd
    precision_prd = np.array(
        [metric_data["prd"][key]["f_beta_data"][1] for key in test_labels]
    )
    precision_prd /= precision_prd[normalization_id]
    recall_prd = np.array(
        [metric_data["prd"][key]["f_beta_data"][0] for key in test_labels]
    )
    recall_prd /= recall_prd[normalization_id]

    # impar
    precision_impar = np.array(
        [metric_data["impar"][key]["precision"] for key in test_labels]
    )
    precision_impar /= precision_impar[normalization_id]
    recall_impar = np.array(
        [metric_data["impar"][key]["recall"] for key in test_labels]
    )
    recall_impar /= recall_impar[normalization_id]

    # fid
    fid = np.array([metric_data["fid"][key]["fid"] for key in test_labels])
    fid /= fid[normalization_id]

    host.plot([], [], "k-", label="quality")
    host.plot([], [], "k--", label="diversity")
    host.plot([], [], ":", label="FID", color=COLOR_PURPLE)
    host.scatter([], [], marker="^", color="r", label="FTI")
    host.scatter([], [], marker="s", color="g", label="PRD")
    host.plot(xticks, precision_prd, "gs-")
    host.plot(xticks, recall_prd, "gs--")
    host.scatter([], [], marker="o", color="b", label="IMPAR")
    host.plot(xticks, precision_impar, "bo-")
    host.plot(xticks, recall_impar, "bo--")

    par1.plot(xticks, fid, ":", color=COLOR_PURPLE)
    host.plot(xticks, precision_fti, "r^-")
    host.plot(xticks, recall_fti, "r^--")

    host.set_xticks(xticks)
    host.set_xticklabels(xlabels)
    par1.set_xticks(xticks)
    par1.set_xticklabels(xlabels)

    host.set_ylim(0, 1.2)
    par1.set_ylim(*fid_ylim)

    xoffset = 0.05
    par1.yaxis.set_label_coords(1.0 + xoffset, 1.06)

    host.set_title(title)
    host.legend(loc=3)

    if not has_host:
        file = os.path.join(root, filename)
        plt.tight_layout()
        plt.savefig(file, dpi=dpi)
        plt.close()


def plot_sampling(
    host,
    root,
    metric_data,
    dataset,
    title,
    xlabel,
    filename,
    dpi=600,
    plot_quality=True,
    plot_diversity=False,
    **kwargs,
):
    has_host = True if host else False
    if not has_host:
        fig, host = plt.subplots()
        fig.subplots_adjust(right=0.76, left=0.08)

    par1 = host.twinx()
    par2 = host.twinx()

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # Offset the right spine of par2. The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.13))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(par2)
    # Second, show the right spine.
    par2.spines["right"].set_visible(True)

    host.set_xlabel(xlabel)
    par1.set_ylabel("FID", rotation="horizontal")
    par2.set_ylabel("IS", rotation="horizontal")

    host.yaxis.label.set_color("k")
    par1.yaxis.label.set_color(COLOR_PURPLE)
    par2.yaxis.label.set_color("c")

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis="y", colors="k", **tkw)
    par1.tick_params(axis="y", colors=COLOR_PURPLE, **tkw)
    par2.tick_params(axis="y", colors="c", **tkw)
    host.tick_params(axis="x", **tkw)

    keys = metric_data["fid"]["samplesize"].unique()

    # fti
    group = metric_data["fti"].groupby(["type", "samplesize"])
    means = group.mean().reset_index()
    mins = group.min().reset_index()
    maxs = group.max().reset_index()

    precision_fti = means.loc[means["type"] == "quality"]["value"].to_numpy()
    precision_fti_mins = (
        mins.loc[means["type"] == "quality"]["value"].to_numpy() / precision_fti[0]
    )
    precision_fti_maxs = (
        maxs.loc[means["type"] == "quality"]["value"].to_numpy() / precision_fti[0]
    )
    precision_fti /= precision_fti[0]

    recall_fti = means.loc[means["type"] == "diversity"]["value"].to_numpy()
    recall_fti_mins = (
        mins.loc[means["type"] == "diversity"]["value"].to_numpy() / recall_fti[0]
    )
    recall_fti_maxs = (
        maxs.loc[means["type"] == "diversity"]["value"].to_numpy() / recall_fti[0]
    )
    recall_fti /= recall_fti[0]

    # prd
    group = metric_data["prd"].groupby(["type", "samplesize"])
    means = group.mean().reset_index()
    mins = group.min().reset_index()
    maxs = group.max().reset_index()

    precision_prd = means.loc[means["type"] == "quality"]["value"].to_numpy()
    precision_prd_mins = (
        mins.loc[means["type"] == "quality"]["value"].to_numpy() / precision_prd[0]
    )
    precision_prd_maxs = (
        maxs.loc[means["type"] == "quality"]["value"].to_numpy() / precision_prd[0]
    )
    precision_prd /= precision_prd[0]

    recall_prd = means.loc[means["type"] == "diversity"]["value"].to_numpy()
    recall_prd_mins = (
        mins.loc[means["type"] == "diversity"]["value"].to_numpy() / recall_prd[0]
    )
    recall_prd_maxs = (
        maxs.loc[means["type"] == "diversity"]["value"].to_numpy() / recall_prd[0]
    )
    recall_prd /= recall_prd[0]

    # impar
    group = metric_data["impar"].groupby(["type", "samplesize"])
    means = group.mean().reset_index()
    mins = group.min().reset_index()
    maxs = group.max().reset_index()

    precision_impar = means.loc[means["type"] == "quality"]["value"].to_numpy()
    precision_impar_mins = (
        mins.loc[means["type"] == "quality"]["value"].to_numpy() / precision_impar[0]
    )
    precision_impar_maxs = (
        maxs.loc[means["type"] == "quality"]["value"].to_numpy() / precision_impar[0]
    )
    precision_impar /= precision_impar[0]

    recall_impar = means.loc[means["type"] == "diversity"]["value"].to_numpy()
    recall_impar_mins = (
        mins.loc[means["type"] == "diversity"]["value"].to_numpy() / recall_impar[0]
    )
    recall_impar_maxs = (
        maxs.loc[means["type"] == "diversity"]["value"].to_numpy() / recall_impar[0]
    )
    recall_impar /= recall_impar[0]

    # is
    group = metric_data["inception_score"].groupby("samplesize")
    inception_score = group.mean()["value"].to_numpy()
    inception_score_mins = group.min()["value"].to_numpy() / inception_score[0]
    inception_score_maxs = group.max()["value"].to_numpy() / inception_score[0]
    inception_score /= inception_score[0]

    # fid
    group = metric_data["fid"].groupby("samplesize")
    fid = group.mean()["value"].to_numpy()
    fid_mins = group.min()["value"].to_numpy() / fid[0]
    fid_maxs = group.max()["value"].to_numpy() / fid[0]
    fid /= fid[0]

    xticks = keys

    def plot_error(ax, xticks, mins, maxs, color):
        ax.fill_between(xticks, mins, maxs, color=color, **{"alpha": 0.2})


    host.plot([], [], ":", label="FID", color=COLOR_PURPLE)
    host.plot([], [], "c-.", label="IS")
    par2.plot(xticks, inception_score, "c-.", label="IS")
    plot_error(par2, xticks, inception_score_mins, inception_score_maxs, "c")
    host.plot([], [], color="r", label="FTI")

    host.plot([], [], color="g", label="PRD")
    if plot_quality:
        host.plot(xticks, precision_prd, color="g")
        plot_error(host, xticks, precision_prd_mins, precision_prd_maxs, "g")
    elif plot_diversity:
        host.plot(xticks, recall_prd, "g")
        plot_error(host, xticks, recall_prd_mins, recall_prd_maxs, "g")

    host.plot([], [], color="b", label="IMPAR")
    if plot_quality:
        host.plot(xticks, precision_impar, color="b")
        plot_error(host, xticks, precision_impar_mins, precision_impar_maxs, "b")
    elif plot_diversity:
        host.plot(xticks, recall_impar, "b")
        plot_error(host, xticks, recall_impar_mins, recall_impar_maxs, "b")
    
    par1.plot(xticks, fid, ":", color=COLOR_PURPLE)
    plot_error(par1, xticks, fid_mins, fid_maxs, COLOR_PURPLE)
    
    if plot_quality:
        host.plot(xticks, precision_fti, color="r")
        plot_error(host, xticks, precision_fti_mins, precision_fti_maxs, "r")
    elif plot_diversity:
        host.plot(xticks, recall_fti, color="r")
        plot_error(host, xticks, recall_fti_mins, recall_fti_maxs, "r")

    par1.set_ylim((0.0, fid.max()))

    xoffset = 0.05
    par1.yaxis.set_label_coords(1.0 + xoffset, 1.06)
    par2.yaxis.set_label_coords(1.13 + xoffset, 1.06)

    host.set_title(title)
    host.legend(loc=3)

    host.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(xticks))
    host.xaxis.set_major_locator(
        matplotlib.ticker.FixedLocator([0, 2000, 4000, 6000, 8000, 10000])
    )
    host.tick_params(axis="x", which="both", bottom=True)
    host.set_xticklabels([0, 2000, 4000, 6000, 8000, 10000])

    if not has_host:
        file = os.path.join(root, filename)
        plt.tight_layout()
        plt.savefig(file, dpi=dpi)
        plt.close()


class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=Formatter)

    parser.add_argument(
        "-root",
        type=str,
        required=True,
        default=None,
        help="Root path to search for data files and write visualizations to.",
    )
    parser.add_argument(
        "-plot_type",
        type=str,
        choices=[
            "noise",
            "mode_dropping",
            "mode_inventing",
            "sampling"
        ],
        required=True,
        default=None,
        help="Type of plot.",
    )
    parser.add_argument(
        "-min_sample_size",
        type=int,
        default=0,
        help="Minimum sample size to visualize.",
    )
    parser.add_argument(
        "-max_sample_size",
        type=int,
        default=sys.maxsize,
        help="Maximum sample size to visualize.",
    )
    parser.add_argument(
        "-plot_per_dataset",
        action="store_true",
        help="If set, create one plot file per dataset instead if possible.",
    )

    args = parser.parse_args()
    plot_type = args.plot_type

    metrics = ["inception_score", "fid", "impar", "prd", "fti"]
    datasets = ["fashion_mnist", "cifar10", "cifar100"]

    metric_to_title = {
        "inception_score": "Inception Score",
        "fid": "Fréchet Inception Distance",
        "impar": "Improved Precision & Recall",
        "prd": "Precision and Recall for Distributions",
        "fti": "Fuzzy Topology Impact",
    }
    metric_to_network = {
        "inception_score": "inception-v3-final",
        "fid": "inception-v3",
        "impar": "vgg-16",
        "prd": "inception-v3",
        "fti": "inception-v3",
    }
    metric_has_swapped = {
        "inception_score": False,
        "fid": False,
        "impar": False,
        "prd": False,
        "fti": True,
    }
    metric_npz_keys = {
        "inception_score": ["inception_score"],
        "fid": ["fid"],
        "impar": ["precision", "recall"],
        "prd": ["f_beta_data"],
        "fti": ["impact"],
    }
    metric_filename_extra_tag = {
        "inception_score": "[test]",
        "fid": None,
        "impar": None,
        "prd": None,
        "fti": None,
    }
    dataset_to_title = {
        "fashion_mnist": "Fashion-MNIST",
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
    }

    if plot_type == "noise":
        dataset_plot_settings = {
            "fashion_mnist": {
                "noise_type": "gaussian",
                "noise_amounts": [0.0001, 0.001, 0.01, 0.1],
                "xlabel": "added Gaussian noise",
                "title": "Fashion-MNIST",
                "fid_ylim": (0.0, 350.0),
            },
            "cifar10": {
                "noise_type": "gaussian",
                "noise_amounts": [0.0001, 0.001, 0.01, 0.1],
                "xlabel": "added Gaussian noise",
                "title": "CIFAR-10",
                "fid_ylim": (0.0, 200.0),
            },
            "cifar100": {
                "noise_type": "gaussian",
                "noise_amounts": [0.0001, 0.001, 0.01, 0.1],
                "xlabel": "added Gaussian noise",
                "title": "CIFAR-100",
                "fid_ylim": (0.0, 150.0),
            },
        }
        if args.plot_per_dataset:
            figs = []
            axes = []
            for _ in range(len(datasets)):
                fig, ax = plt.subplots(figsize=(5, 3.75))
                figs.append(fig)
                axes.append(ax)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
            fig.subplots_adjust(right=0.76, left=0.08)
            axes = axes.flatten()
        for i, dataset in enumerate(datasets):
            print(f"Loading {dataset}")
            metric_data = {}
            for metric in metrics:
                swaps = [False] if not metric_has_swapped[metric] else [False, True]
                for swap in swaps:
                    tag = metric_filename_extra_tag[metric]
                    print(
                        f"Loading {metric}>{metric_to_network[metric]}>{dataset}{'>swapped.' if swap else ''}{dataset_plot_settings[dataset]['noise_type']}>{dataset}{tag if tag else ''}"
                    )
                    original_data = load_original(
                        args.root,
                        metric,
                        metric_to_network[metric],
                        dataset,
                        tag=tag,
                        swapped=swap,
                    )
                    data = load_data(
                        args.root,
                        metric,
                        metric_to_network[metric],
                        dataset,
                        dataset_plot_settings[dataset]["noise_type"],
                        tag=tag,
                        swapped=swap,
                    )

                    target = "test_" if not swap else "train_"
                    if metric == "inception_score":
                        target = ""
                    noise_amount_key = f"{target}noise_amount"

                    # Filter noises
                    data = [
                        d
                        for d in data
                        if d[noise_amount_key]
                        in dataset_plot_settings[dataset]["noise_amounts"]
                    ]

                    metric_key = metric if not swap else f"{metric}.swapped"

                    metric_data[metric_key] = {
                        d[noise_amount_key][()]: {
                            data_key: d[data_key]
                            for data_key in metric_npz_keys[metric]
                        }
                        for d in data
                    }
                    metric_data[metric_key].update(
                        {
                            0: {
                                data_key: original_data[data_key]
                                for data_key in metric_npz_keys[metric]
                            }
                        }
                    )
            images = load_images(
                args.root,
                dataset_plot_settings[dataset]["noise_type"],
                dataset_plot_settings[dataset]["noise_amounts"],
                dataset,
            )
            noise(
                axes[i],
                fig,
                i,
                args.root,
                metric_data,
                dataset,
                images=images,
                **dataset_plot_settings[dataset],
            )

        if args.plot_per_dataset:
            for fig, dataset in zip(figs, datasets):
                fig.tight_layout()
                file = os.path.join(args.root, f"gaussian_noise-{dataset}.pdf")
                fig.savefig(file, dpi=600, bbox_inches="tight")
                plt.close(fig)
        else:
            plt.tight_layout()
            file = os.path.join(args.root, "gaussian_noise.pdf")
            plt.savefig(file, dpi=600, bbox_inches="tight")
            plt.close()
    elif plot_type == "mode_dropping":
        current_metrics = ["fid", "impar", "prd", "fti"]
        dataset_plot_settings = {
            "fashion_mnist": {
                "filename": "fashion-mnist_mode_dropping.pdf",
                "xlabel": "# dropped classes",
                "title": "Fashion-MNIST",
                "fid_ylim": (0.0, 40.0),
                "legend_loc": "center left",
            },
            "cifar10": {
                "filename": "cifar10_mode_dropping.pdf",
                "xlabel": "# dropped classes",
                "title": "CIFAR-10",
                "fid_ylim": (0.0, 8.0),
                "legend_loc": "lower center",
            },
            "cifar100": {
                "filename": "cifar100_mode_dropping.pdf",
                "xlabel": "# dropped classes",
                "title": "CIFAR-100",
                "fid_ylim": (0.0, 4.0),
                "legend_loc": "lower center",
            },
        }
        if args.plot_per_dataset:
            figs = []
            axes = []
            for _ in range(len(datasets)):
                fig, ax = plt.subplots(figsize=(5, 3.75))
                figs.append(fig)
                axes.append(ax)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
            fig.subplots_adjust(right=0.76, left=0.08)
            axes = axes.flatten()
        for i, dataset in enumerate(datasets):
            print(f"Loading {dataset}")
            metric_data = OrderedDict()
            for metric in current_metrics:
                swaps = [False] if not metric_has_swapped[metric] else [False, True]
                for swap in swaps:
                    tag = metric_filename_extra_tag[metric]
                    print(
                        f"Loading {metric}>{metric_to_network[metric]}>{dataset}>{'swapped.' if swap else ''}mode_dropping>{dataset}{tag if tag else ''}"
                    )
                    data = load_data(
                        args.root,
                        metric,
                        metric_to_network[metric],
                        dataset,
                        "mode_dropping",
                        tag=tag,
                        swapped=swap,
                    )

                    target = "test_" if not swap else "train_"
                    other_target = "test_" if swap else "train_"
                    label_key = f"allowed_{target}labels"
                    other_label_key = f"allowed_{other_target}labels"

                    metric_key = metric if not swap else f"{metric}.swapped"

                    for d in reversed(data):
                        labels = d[label_key]
                        other_labels = d[other_label_key]
                        entry = metric_data.setdefault(metric_key, {})
                        entry.update({(labels.min(), labels.max()): d})
            plot_mode_dropping(
                axes[i],
                args.root,
                metric_data,
                dataset,
                **dataset_plot_settings[dataset],
            )

        if args.plot_per_dataset:
            for fig, dataset in zip(figs, datasets):
                fig.tight_layout()
                file = os.path.join(args.root, f"mode_dropping-{dataset}.pdf")
                fig.savefig(file, dpi=600, bbox_inches="tight")
                plt.close(fig)
        else:
            plt.tight_layout()
            file = os.path.join(args.root, "mode_dropping.pdf")
            plt.savefig(file, dpi=600, bbox_inches="tight")
            plt.close()
    elif plot_type == "mode_inventing":
        current_metrics = ["fid", "impar", "prd", "fti"]
        dataset_plot_settings = {
            "fashion_mnist": {
                "filename": "_fashion-mnist_mode_inventing.pdf",
                "xlabel": "class range",
                "title": "Fashion-MNIST",
                "fid_ylim": (0.0, 25.0),
            },
            "cifar10": {
                "filename": "_cifar10_mode_inventing.pdf",
                "xlabel": "class range",
                "title": "CIFAR-10",
                "fid_ylim": (0.0, 15.0),
            },
            "cifar100": {
                "filename": "_cifar100_mode_inventing.pdf",
                "xlabel": "class range",
                "title": "CIFAR-100",
                "fid_ylim": (0.0, 8.0),
            },
        }
        if args.plot_per_dataset:
            figs = []
            axes = []
            for _ in range(len(datasets)):
                fig, ax = plt.subplots(figsize=(5, 3.75))
                figs.append(fig)
                axes.append(ax)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
            fig.subplots_adjust(right=0.76, left=0.08)
            axes = axes.flatten()
        for i, dataset in enumerate(datasets):
            print(f"Loading {dataset}")
            metric_data = OrderedDict()
            for metric in current_metrics:
                swaps = [False] if not metric_has_swapped[metric] else [False, True]
                for swap in swaps:
                    tag = metric_filename_extra_tag[metric]
                    print(
                        f"Loading {metric}>{metric_to_network[metric]}>{dataset}>{'swapped.' if swap else ''}mode_inventing>{dataset}{tag if tag else ''}"
                    )
                    data = load_data(
                        args.root,
                        metric,
                        metric_to_network[metric],
                        dataset,
                        "mode_inventing",
                        tag=tag,
                        swapped=swap,
                    )

                    target = "test_" if not swap else "train_"
                    other_target = "test_" if swap else "train_"
                    label_key = f"allowed_{target}labels"
                    other_label_key = f"allowed_{other_target}labels"

                    metric_key = metric if not swap else f"{metric}.swapped"

                    for d in data:
                        labels = d[label_key]
                        other_labels = d[other_label_key]
                        entry = metric_data.setdefault(metric_key, {})
                        entry.update({(labels.min(), labels.max()): d})
            plot_mode_inventing(
                axes[i],
                args.root,
                metric_data,
                dataset,
                **dataset_plot_settings[dataset],
            )

        if args.plot_per_dataset:
            for fig, dataset in zip(figs, datasets):
                fig.tight_layout()
                file = os.path.join(args.root, f"mode_inventing-{dataset}.pdf")
                fig.savefig(file, dpi=600, bbox_inches="tight")
                plt.close(fig)
        else:
            plt.tight_layout()
            file = os.path.join(args.root, "mode_inventing.pdf")
            plt.savefig(file, dpi=600, bbox_inches="tight")
            plt.close()
    elif plot_type == "sampling":
        dataset_plot_settings = {
            "fashion_mnist": {
                "filename": "fashion-mnist_sampling.pdf",
                "xlabel": "class",
                "title": "Fashion-MNIST",
            },
            "cifar10": {
                "filename": "cifar10_sampling.pdf",
                "xlabel": "class",
                "title": "CIFAR-10",
            },
            "cifar100": {
                "filename": "cifar100_sampling.pdf",
                "xlabel": "class",
                "title": "CIFAR-100",
            },
        }

        def to_pandas_dataframe(data, metric):
            if metric == "inception_score":
                data = data[0]
                columns = ["samplesize", "Inception Score"]
                sample_sizes = []
                inception_scores = []
                for size_key in sorted(data.keys()):
                    samples = data[size_key]
                    for sample in samples:
                        sample_sizes.append(size_key)
                        inception_scores.append(sample["inception_score"][0])

                sample_sizes = np.array(sample_sizes)[:, None]
                inception_scores = np.array(inception_scores)[:, None]
                data = np.hstack([sample_sizes, inception_scores])
            elif metric == "fid":
                data = data[0]
                columns = ["samplesize", "Fréchet Inception Distance"]
                sample_sizes = []
                fids = []
                for size_key in sorted(data.keys()):
                    samples = data[size_key]
                    for sample in samples:
                        sample_sizes.append(size_key)
                        fids.append(sample["fid"])

                sample_sizes = np.array(sample_sizes)[:, None]
                fids = np.array(fids)[:, None]
                data = np.hstack([sample_sizes, fids])
            elif metric == "impar":
                data = data[0]
                columns = ["samplesize", "quality", "diversity"]
                sample_sizes = []
                qualities = []
                diversities = []
                for size_key in sorted(data.keys()):
                    samples = data[size_key]
                    for sample in samples:
                        sample_sizes.append(size_key)
                        qualities.append(sample["precision"][0])
                        diversities.append(sample["recall"][0])

                sample_sizes = np.array(sample_sizes)[:, None]
                qualities = np.array(qualities)[:, None]
                diversities = np.array(diversities)[:, None]
                data = np.hstack([sample_sizes, qualities, diversities])
            elif metric == "prd":
                data = data[0]
                columns = ["samplesize", "quality", "diversity"]
                sample_sizes = []
                qualities = []
                diversities = []
                for size_key in sorted(data.keys()):
                    samples = data[size_key]
                    for sample in samples:
                        sample_sizes.append(size_key)
                        qualities.append(sample["f_beta_data"][1])
                        diversities.append(sample["f_beta_data"][0])

                sample_sizes = np.array(sample_sizes)[:, None]
                qualities = np.array(qualities)[:, None]
                diversities = np.array(diversities)[:, None]
                data = np.hstack([sample_sizes, qualities, diversities])
            elif metric == "fti":
                columns = ["samplesize", "quality", "diversity"]
                sample_sizes = []
                qualities = []
                diversities = []
                for size_key in sorted(data[0].keys()):
                    samples = data[0][size_key]
                    for sample in samples:
                        sample_sizes.append(size_key)
                        qualities.append(sample["impact"])
                for size_key in sorted(data[1].keys()):
                    samples = data[1][size_key]
                    for sample in samples:
                        diversities.append(sample["impact"])

                sample_sizes = np.array(sample_sizes)[:, None]
                qualities = np.array(qualities)[:, None]
                diversities = np.array(diversities)[:, None]
                data = np.hstack([sample_sizes, qualities, diversities])

            df = pd.DataFrame(data=data, columns=columns)
            df = pd.melt(df, id_vars="samplesize", var_name="type", value_name="value")
            df["samplesize"] = pd.to_numeric(df["samplesize"], downcast="integer")
            return df

        if args.plot_per_dataset:
            figs = []
            axes = []
            for _ in range(len(datasets)):
                fig, ax = plt.subplots(figsize=(5, 3.75))
                figs.append(fig)
                axes.append(ax)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
            fig.subplots_adjust(right=0.76, left=0.08)
            axes = axes.flatten()
        for i, dataset in enumerate(datasets):
            print(f"Loading {dataset}")
            metric_data = OrderedDict()
            for metric in metrics:
                swaps = [False] if not metric_has_swapped[metric] else [False, True]
                raw = []
                for swap in swaps:
                    tag = metric_filename_extra_tag[metric]
                    print(
                        f"Loading {metric}>{metric_to_network[metric]}>{dataset}>{'swapped.' if swap else ''}sample_test>{dataset}{tag if tag else ''}"
                    )
                    data = load_data(
                        args.root,
                        metric,
                        metric_to_network[metric],
                        dataset,
                        "sample_test",
                        tag=tag,
                        swapped=swap,
                    )

                    target = "_test" if not swap else "_train"
                    if metric == "inception_score":
                        target = ""
                    sample_key = f"sample{target}"

                    metric_key = metric if not swap else f"{metric}.swapped"

                    # Filter sizes
                    data = [
                        d
                        for d in data
                        if args.min_sample_size <= d[sample_key] <= args.max_sample_size
                    ]

                    group = OrderedDict()
                    for d in data:
                        size = d[sample_key][()]
                        group.setdefault(size, []).append(d)
                    raw.append(group)

                df = to_pandas_dataframe(raw, metric)
                metric_data.update({metric: df})
            plot_sampling(
                axes[i],
                args.root,
                metric_data,
                dataset,
                **dataset_plot_settings[dataset],
            )

        if args.plot_per_dataset:
            for fig, dataset in zip(figs, datasets):
                fig.tight_layout()
                file = os.path.join(args.root, f"sample-singleplot-{dataset}.pdf")
                fig.savefig(file, dpi=600, bbox_inches="tight")
                plt.close(fig)
        else:
            plt.tight_layout()
            file = os.path.join(args.root, "sample-singleplot.pdf")
            plt.savefig(file, dpi=600, bbox_inches="tight")
            plt.close()
