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

from kedro.pipeline import Pipeline, node

from .nodes import b_ascent_analysis, b_descent_analysis, lambda_analysis, truncate_dataset, split_train_pool, compute_gaussian_kernel, al_performances, pl_performances


def create_pipeline(**kwargs):
    return Pipeline(
        [
            #node(
            #    func=split_train_pool,
            #    inputs=dict(
            #        y_train_full="y_train_full",
            #        n_init="params:N_INIT"
            #        ),
            #    outputs=["full_id", "train_id", "pool_id"],
            #    tags=["sampling"]
            #),
            node(
                func=truncate_dataset,
                inputs=dict(
                    X_train_full="X_train_full",
                    y_train_full="y_train_full",
                    size="params:SIZE_ANALYSIS"
                    ),
                outputs=["X_train_trunc", "y_train_trunc"],
                tags=["pre_sampling"]
            ),
            node(
                func=compute_gaussian_kernel,
                inputs=dict(
                    X="X_train_trunc"
                    ),
                outputs="K_FIXE",
                tags=["pre_sampling"]
            ),
            node(
                func=al_performances,
                inputs=dict(
                    bs="params:BATCH_SEQ",
                    budget="params:BUDGET",
                    n_simu="params:N_SIMULATIONS",
                    X_train_full="X_train_trunc",
                    y_train_full="y_train_trunc",
                    X_test="X_test",
                    y_test="y_test",
                    K_FIXE="K_FIXE",
                    n_init="params:N_INIT",
                    lam = "params:LAMBDA"
                    ),
                outputs="al_perfs",
                tags=["sampling", "active_sampling"]
            ),
            node(
                func=lambda_analysis,
                inputs=dict(
                    b="params:BATCH_SIZE",
                    budget="params:BUDGET",
                    n_simu="params:N_SIMULATIONS",
                    X_train_full="X_train_trunc",
                    y_train_full="y_train_trunc",
                    X_test="X_test",
                    y_test="y_test",
                    n_init="params:N_INIT",
                    K_FIXE="K_FIXE"
                    ),
                outputs="al_lam_perfs",
                tags=["sampling", "active_lambda_sampling"]
            ),
            node(
                func=pl_performances,
                inputs=dict(
                    bs="params:BATCH_SEQ",
                    budget="params:BUDGET",
                    n_simu="params:N_SIMULATIONS",
                    X_train_full="X_train_trunc",
                    y_train_full="y_train_trunc",
                    X_test="X_test",
                    y_test="y_test",
                    n_init="params:N_INIT"
                    ),
                outputs="pl_perfs",
                tags=["sampling", "passive_sampling"]
            ),
            node(
                func=b_descent_analysis,
                inputs=dict(
                    budget="params:BUDGET",
                    n_simu="params:N_SIMULATIONS",
                    X_train_full="X_train_trunc",
                    y_train_full="y_train_trunc",
                    X_test="X_test",
                    y_test="y_test",
                    n_init="params:N_INIT",
                    b_descent_size="params:BATCH_DESCENT_SIZE",
                    b_rate="params:BATCH_DESCENT_RATE",
                    K_FIXE="K_FIXE"
                    ),
                outputs=["b_descent_perfs", "bs_descent"],
                tags=["sampling", "active_descent_sampling"]
            ),
            node(
                func=b_ascent_analysis,
                inputs=dict(
                    budget="params:BUDGET",
                    n_simu="params:N_SIMULATIONS",
                    X_train_full="X_train_trunc",
                    y_train_full="y_train_trunc",
                    X_test="X_test",
                    y_test="y_test",
                    n_init="params:N_INIT",
                    b_ascent_size="params:BATCH_ASCENT_SIZE",
                    b_rate="params:BATCH_ASCENT_RATE",
                    K_FIXE="K_FIXE"
                    ),
                outputs=["b_ascent_perfs", "bs_ascent"],
                tags=["sampling", "active_descent_sampling"]
            ),
        ]
    )