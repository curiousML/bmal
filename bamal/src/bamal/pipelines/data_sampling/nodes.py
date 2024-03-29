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

def al_performances(bs, budget, n_simu, X_train_full, y_train_full, X_test, y_test, K_FIXE = None, n_init = 0, lam = "normal"):
    perfs_dict = dict()
    lam_dict = dict()

    for b in bs:
        
        lam_dict["small"] = np.sqrt(b)/2
        lam_dict["normal"] = None
        lam_dict["big"] = b**5

        perfs_dict[b] = []
        for s in range(n_simu):
            if s%5 == 0:
                logging.info(f"active learning, b : {b}, s : {s}")
            # refaire le sampling si pas de représentation de toutes les classes
            if n_init > 0:
                full_id, train_id, pool_id = split_train_pool(y_train_full, n_init = n_init)
            else:
                full_id, train_id, pool_id = split_train_pool(y_train_full, n_init = b)
            al = ActiveLearning(X_train_full, y_train_full, full_id, train_id, pool_id, b, K_FIXE, lam=lam_dict[lam]) #lam_dict[lam]
            perfs, batches = al.run(X_test, y_test, (budget-n_init)//b)
            perfs_dict[b].append(perfs)
    return perfs_dict


def pl_performances(bs, budget, n_simu, X_train_full, y_train_full, X_test, y_test, n_init = 0):
    perfs_dict = dict()
    for b in bs:
        perfs_dict[b] = []
        for s in range(n_simu):
            if s%5 == 0:
                logging.info(f"passive learning, b : {b}, s : {s}")
            # refaire le sampling si pas de représentation de toutes les classes
            if n_init > 0:
                full_id, train_id, pool_id = split_train_pool(y_train_full, n_init = n_init)
            else:
                full_id, train_id, pool_id = split_train_pool(y_train_full, n_init = b)
            pl = PassiveLearning(X_train_full, y_train_full, full_id, train_id, pool_id, b)
            perfs_dict[b].append(pl.run(X_test, y_test, (budget-n_init)//b))
    return perfs_dict


def lambda_analysis(b, budget, n_simu, X_train_full, y_train_full, X_test, y_test, n_init, K_FIXE = None):
    perfs_dict = dict()
    lam = dict()
    lam["small"] = np.sqrt(b)/2
    lam["normal"] = b
    lam["big"] = b**5

    for m in lam:
        perfs_dict[m] = []

        for s in range(n_simu):
            if s%5 == 0:
                logging.info(f"active learning, b : {b}, s : {s}, lam : {m}")
            # refaire le sampling si pas de représentation de toutes les classes
            full_id, train_id, pool_id = split_train_pool(y_train_full, n_init = b)
            #while np.len(np.unique(y_train_full[train_id]))
            al = ActiveLearning(X_train_full, y_train_full, full_id, train_id, pool_id, b, K_FIXE, lam = lam[m])
            perfs, batches = al.run(X_test, y_test, budget//b)
            perfs_dict[m].append(perfs)

    return perfs_dict

# TODO: updates these functions
# TODO: try other functions of "b"
def b_descent_analysis(budget, n_simu, X_train_full, y_train_full, X_test, y_test, n_init, b_descent_size, b_rate = 0, K_FIXE = None):
    # descent analysis (TODO: more exotic way)
    for N in np.arange(b_descent_size, budget, b_descent_size):
        b_descent = np.concatenate(([0], np.arange(N, 0, -b_descent_size)))
        x_candidate = np.cumsum(b_descent)+n_init
        if x_candidate[-1] >= budget:
            x = x_candidate
            break

    if b_rate > 0:
        times = len(x)+round(b_rate*10)
    else:
        times = len(x)-1

    perfs = []
    b_init = N
    for s in range(n_simu):
        if s%5 == 0:
            logging.info(f"nb simulations : {s}")
        # refaire le sampling si pas de représentation de toutes les classes
        full_id, train_id, pool_id = split_train_pool(y_train_full, n_init = n_init)
        #while np.len(np.unique(y_train_full[train_id]))
        al = ActiveLearning(X_train_full, y_train_full, full_id, train_id, pool_id, b_init, K_FIXE)
        perf, bs = al.run(X_test, y_test, times, b_descent=b_descent_size, b_rate=b_rate)
        perfs.append(perf)
    logging.info(f"bs : {bs}")
    logging.info(f"perfs : {perfs[0]}")
    return perfs, bs

def b_ascent_analysis(budget, n_simu, X_train_full, y_train_full, X_test, y_test, n_init, b_ascent_size, b_rate = 0, K_FIXE = None):
    # ascent analysis (TODO: more exotic way)
    for N in np.arange(b_ascent_size, budget, b_ascent_size):
        b_ascent = np.arange(0, N, b_ascent_size)
        x_candidate = np.cumsum(b_ascent)+n_init
        if x_candidate[-1] >= budget:
            x = x_candidate
            break
    
    if b_rate < 0:
        times = len(x)+round(b_rate*10)
    else:
        times = len(x)

    perfs = []
    b_init = b_ascent_size
    for s in range(n_simu):
        if s%5 == 0:
            logging.info(f"nb simulations : {s}")
        # refaire le sampling si pas de représentation de toutes les classes
        full_id, train_id, pool_id = split_train_pool(y_train_full, n_init = n_init)
        #while np.len(np.unique(y_train_full[train_id]))
        al = ActiveLearning(X_train_full, y_train_full, full_id, train_id, pool_id, b_init, K_FIXE)
        perf, bs = al.run(X_test, y_test, times, b_descent=-b_ascent_size, b_rate=b_rate)
        perfs.append(perf)
    logging.info(f"bs : {bs}")
    logging.info(f"perfs : {perfs[0]}")
    return perfs, bs