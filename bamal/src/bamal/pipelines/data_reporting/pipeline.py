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

from .nodes import plot_b_descent_line_line, plot_line_line, plot_line_box, plot_multiple_line_box, plot_batch_line_line, plot_lambda_line_line


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=plot_line_line,
                inputs=dict(
                    pl_perfs="pl_perfs",
                    al_perfs="al_perfs",
                    bs="params:BATCH_SEQ",
                    budget="params:BUDGET",
                    n_init="params:N_INIT"
                    ),
                outputs="line_line",
                tags=["reporting"]
            ),
            node(
                func=plot_line_box,
                inputs=dict(
                    pl_perfs="pl_perfs",
                    al_perfs="al_perfs",
                    budget="params:BUDGET",
                    b_analysis="params:BATCH_ANALYSIS",
                    n_init="params:N_INIT"
                    ),
                outputs="line_box",
                tags=["reporting"]
            ),
            node(
                func=plot_batch_line_line,
                inputs=dict(
                    Bs="params:BUDGET_SEQ",
                    bs="params:BATCH_SEQ",
                    perfs_dict="al_perfs",
                    n_init="params:N_INIT"
                    ),
                outputs="batch_line_line",
                tags=["reporting"]
            ),
            node(
                func=plot_lambda_line_line,
                inputs=dict(
                    pl_perfs="pl_perfs",
                    al_lam_perfs="al_lam_perfs",
                    b="params:BATCH_SIZE",
                    budget="params:BUDGET",
                    n_init="params:N_INIT"
                    ),
                outputs="lambda_line_line",
                tags=["reporting", "lambda_analysis"]
            ),
            node(
                func=plot_b_descent_line_line,
                inputs=dict(
                    pl_perfs="pl_perfs",
                    al_perfs="al_perfs",
                    b_descent_perfs="b_descent_perfs",
                    b_ascent_perfs="b_ascent_perfs",
                    b="params:BATCH_SIZE",
                    bs_descent="bs_descent", 
                    bs_ascent ="bs_ascent",
                    budget="params:BUDGET",
                    n_init="params:N_INIT",
                    descent_rate="params:BATCH_DESCENT_RATE", 
                    ascent_rate ="params:BATCH_ASCENT_RATE"
                    ),
                outputs="b_descent_line_line",
                tags=["reporting", "b_descent_analysis"]
            ),
            #node(
            #    func=plot_multiple_line_box,
            #    inputs=dict(
            #        pl_perfs="pl_perfs",
            #        al_perfs="al_perfs",
            #        bs="params:BATCH_SEQ",
            #        budget="params:BUDGET",
            #        n_init="params:N_INIT"
            #        ),
            #    outputs=None,
            #    tags=["reporting"]
            #),
        ]
    )

# see run id : 2021-02-22T17.13.42.763Z