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


def plot_line_line(pl_perf, al_perfs, bs, budget):
    pl_x = np.arange(bs[0], budget+bs[0], bs[0])
    pl_y = np.array(pl_perf).mean(axis = 0)[:len(pl_x)]
    plt.plot(pl_x, pl_y, label = "passive learning", marker = 'o')
    for b in bs:
        x = np.arange(b, budget+b, b)
        perf_mean = np.array(al_perfs[b]).mean(axis = 0)[:len(x)]
        img = plt.plot(x, perf_mean, label = "active learning " + str(b), marker = '+')
    plt.legend()
    #plt.show()
    return None

def plot_line_box(pl_perf, al_perfs, bs, budget, b_analysis):
    x = np.arange(0, budget+b_analysis, b_analysis)+b_analysis
    perfs = pd.DataFrame({
        i:perf for i, perf in enumerate(al_perfs[b_analysis])
    }, index = x).transpose()
    perfs = perfs.iloc[:,:-1]
    
    pl_x = np.arange(bs[0], budget+bs[0], bs[0])
    pl_y = np.array(pl_perf).mean(axis = 0)[:len(pl_x)]
    g1 = sns.boxplot(data = perfs, showfliers=False, color="orange")
    plt.plot(pl_x.astype(str), pl_y, label = "passive learning", marker = 'o')
    g1.set(xticklabels=[])
    g1.set(ylabel="AUC")
    g1.set(xlabel=f"B (0 à {budget})")
    g1.set(title=f"b = {b_analysis}")
    g1.legend()
    plt.show()
    return None
