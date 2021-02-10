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

PLEASE DELETE THIS FILE ONCE YOU START WORKING ON YOUR OWN PROJECT!
"""

from typing import Any, Dict
import logging
import pandas as pd
import numpy as np
from .samplings.active_sampling import ActiveLearning
from .samplings.passive_sampling import PassiveLearning
from scipy.spatial.distance import pdist, squareform


def split_train_pool(y_train_full, n_init = 200):
    """
    returns only pool, train and full ids
    """
    size            =  len(y_train_full)
    full_id         =  np.arange(size)
    init_selection  =  np.random.choice(size, n_init, replace=False)
    train_id        =  init_selection
    pool_id         =  np.delete(full_id, init_selection)
    #logging.info("train_id lengths : " + str(len(train_id)))
    #logging.info("pool_id lengths : " + str(len(pool_id)))
    return full_id, train_id, pool_id

def truncate_dataset(X_train_full, y_train_full, size = 10000):
    return X_train_full[:size], y_train_full[:size]

def compute_gaussian_kernel(X):
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    p = np.median(pairwise_dists ** 2)
    K = np.exp(-pairwise_dists ** 2 / p)
    return K

def al_performances(bs, budget, n_simu, X_train_full, y_train_full, X_test, y_test, K_FIXE = None):
    perfs_dict = dict()
    for b in bs:
        perfs_dict[b] = []
        for s in range(n_simu):
            if s%5 == 0:
                logging.info(f"active learning, b : {b}, s : {s}")
            # refaire le sampling si pas de représentation de toutes les classes
            full_id, train_id, pool_id = split_train_pool(y_train_full, n_init = b)
            #while np.len(np.unique(y_train_full[train_id]))
            al = ActiveLearning(X_train_full, y_train_full, full_id, train_id, pool_id, b, K_FIXE)
            perfs_dict[b].append(al.run(X_test, y_test, budget//b))
    return perfs_dict


def pl_performances(bs, budget, n_simu, X_train_full, y_train_full, X_test, y_test):
    perfs = []
    for s in range(n_simu):
        if s%5 == 0:
            logging.info(f"passive learning, b : {bs[0]}, s : {s}")
        full_id, train_id, pool_id = split_train_pool(y_train_full, n_init = bs[0])
        pl = PassiveLearning(X_train_full, y_train_full, full_id, train_id, pool_id, bs[0])
        perfs.append(pl.run(X_test, y_test, budget//bs[0]))
    return perfs
