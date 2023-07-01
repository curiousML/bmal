# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""
# pylint: disable=invalid-name

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kedro.extras.datasets.matplotlib import MatplotlibWriter
from kedro_mlflow.io.artifacts import MlflowArtifactDataSet
# TODO: there's an error of shifting to fix

def plot_line_line(pl_perfs, al_perfs, bs, budget, n_init):
    pl_x = np.arange(n_init, budget+bs[0], bs[0])
    pl_y = np.array(pl_perfs[bs[0]]).mean(axis = 0)
    n_points = min(len(pl_x), len(pl_y))
    plt.plot(pl_x[:n_points], pl_y[:n_points], label = "passive learning", marker = 'o')
    for b in bs:
        x = np.arange(n_init, budget+b, b)
        perf_mean = np.array(al_perfs[b]).mean(axis = 0)[:len(x)]
        n_points = min(len(x), len(perf_mean))
        plt.plot(x[:n_points], perf_mean[:n_points], label = "active learning " + str(b), marker = '+')
    plt.xlabel("B")
    plt.ylabel("F1-score")
    #plt.title(f"b = {b}")
    plt.legend()
    img = plt.gcf()
    plt.close("all")
    #plot_writer = MatplotlibWriter(
    #    filepath="data/08_reporting/line_line.png"
    #)
    #plot_writer.save(img)
    return img

def plot_line_box(pl_perfs, al_perfs, budget, b_analysis, n_init):
    x = np.arange(n_init, budget+b_analysis, b_analysis)
    n_points = min(len(x), len(al_perfs[b_analysis]))

    perfs = pd.DataFrame({
        i:perf for i, perf in enumerate(al_perfs[b_analysis][:n_points])
    }).transpose()
    perfs = perfs.iloc[:,:-1]
    
    pl_x = np.arange(n_init, budget+b_analysis, b_analysis)
    pl_y = np.array(pl_perfs[b_analysis]).mean(axis = 0)
    n_points = min(len(pl_x), len(pl_y))
    
    g1 = sns.boxplot(data = perfs, showfliers=False, color="orange")
    plt.plot(pl_x[:n_points].astype(str), pl_y[:n_points], label = "passive learning", marker = 'o')
    g1.set(xticklabels=[])
    g1.set(ylabel="F1-score")
    g1.set(xlabel=f"B (0 to {budget})")
    g1.set(title=f"b = {b_analysis}")
    g1.legend()
    img = plt.gcf()
    plt.close("all")
    #plot_writer = MatplotlibWriter(
    #    filepath=f"data/08_reporting/line_box_{b_analysis}.png"
    #)
    #plot_writer.save(img)
    return img

def plot_multiple_line_box(pl_perfs, al_perfs, bs, budget,n_init):
    plots_dict = dict()
    for b in bs:
        plots_dict[b] = plot_line_box(pl_perfs, al_perfs, bs, budget, b)
        plt.close()
    #img.savefig('data/08_reporting/multi_line_box.png')
    plot_writer = MatplotlibWriter(
        filepath="data/08_reporting/multi_line_box"
    )
    plot_writer.save(plots_dict)
    return None

# TODO: readjust the points
def plot_batch_line_line(Bs, bs, perfs_dict, n_init):
    for B in Bs:
        perfs = [np.array(perfs_dict[b]).mean(axis = 0)[B//b] for b in bs]
        plt.plot(np.array(bs).astype(str), perfs, marker = 'o', label = f"Budget : {B}")
    plt.xlabel("b")
    plt.ylabel("F1-score")
    plt.legend()
    img = plt.gcf()
    plt.close("all")
    #plot_writer = MatplotlibWriter(
    #    filepath=f"data/08_reporting/batch_line_line.png"
    #)
    #plot_writer.save(img)
    return img


def plot_lambda_line_line(pl_perfs, al_lam_perfs, b, budget,n_init):
    pl_x = np.arange(n_init, budget+b, b)
    pl_y = np.array(pl_perfs[b]).mean(axis = 0)
    n_points = min(len(pl_x), len(pl_y))
    plt.plot(pl_x[:n_points], pl_y[:n_points], label = "passive learning", marker = 'o')
    for lam in al_lam_perfs:
        x = np.arange(b, budget+b, b)
        perf_mean = np.array(al_lam_perfs[lam]).mean(axis = 0)
        n_points = min(len(x), len(perf_mean))
        plt.plot(x[:n_points], perf_mean[:n_points], label = "active learning, lambda " + lam, marker = '+')
    plt.title(f"b = {b}")
    plt.legend()
    plt.xlabel("B")
    plt.ylabel("F1-score")
    img = plt.gcf()
    plt.close("all")
    #plot_writer = MatplotlibWriter(
    #    filepath="data/08_reporting/lambda_line_line.png"
    #)
    #plot_writer.save(img)
    return img


def plot_b_descent_line_line(pl_perfs, al_perfs, b_descent_perfs, b_ascent_perfs, b, bs_descent, bs_ascent, budget, n_init, descent_rate = 0, ascent_rate = 0):
    pl_x = np.arange(n_init, budget+b, b)
    pl_y = np.array(pl_perfs[b]).mean(axis = 0)
    n_points = min(len(pl_x), len(pl_y))
    plt.plot(pl_x[:n_points], pl_y[:n_points], label = f"static-size PL b = {b}", marker = 'o')
    al_x = np.arange(n_init, budget+b, b)
    al_y = np.array(al_perfs[b]).mean(axis = 0)
    n_points = min(len(al_x), len(al_y))
    plt.plot(al_x[:n_points], al_y[:n_points], label = f"static-size AL, b = {b}", marker = 'o')

    perf_mean = np.array(b_descent_perfs).mean(axis = 0)
    x_descent = np.cumsum(bs_descent)
    plt.plot(x_descent, perf_mean, label = f"decreasing size AL ({bs_descent[1]} to {bs_descent[-1]}", marker = 'o')
    
    # ascent analysis (TODO: more exotic way)
    perf_mean = np.array(b_ascent_perfs).mean(axis = 0)
    x_ascent = np.cumsum(bs_ascent)
    plt.plot(x_ascent, perf_mean, label = f"increasing size AL ({bs_ascent[1]} to {bs_ascent[-1]}", marker = 'o')
    plt.xlabel("B")
    plt.ylabel("F1-score")
    #plt.title(f"b = {b}")
    plt.legend()
    img = plt.gcf()
    plt.close("all")
    #plot_writer = MatplotlibWriter(
    #    filepath="data/08_reporting/b_descent_line_line.png"
    #)
    #plot_writer.save(img)
    return img

#def plot_batch_box_box():
#
#    return None
#
#budget_middle = 100
#budget_final = BUDGET
#perfs_middle = [np.array(perfs_dict[b]).mean(axis = 0)[budget_middle//b] for b in bs]
#perfs_final = [np.array(perfs_dict[b]).mean(axis = 0)[budget_final//b] for b in bs]
#
#df_middle = pd.DataFrame({
#    b : np.array(perfs_dict[b])[:,budget_middle//b] for b in bs
#})
#sns.boxplot(data = df_middle, showfliers=False, color="steelblue")
#
#df_final = pd.DataFrame({
#    b : np.array(perfs_dict[b])[:,budget_final//b] for b in bs
#})
#sns.boxplot(data = df_final, showfliers=False, color="orange")
#
#plt.plot(np.array(bs).astype(str), perfs_middle, marker = 'o', label = f"label {budget_middle + len(init_selection)}")
#plt.plot(np.array(bs).astype(str), perfs_final, marker = 'o', label = f"label {budget_final + len(init_selection)}")
#
#plt.xlabel("b")
#plt.ylabel("perf")
#plt.legend()
#plt.show()