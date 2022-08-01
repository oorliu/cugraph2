# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import gc

import pytest
import networkx as nx
import cugraph

from cugraph.experimental.datasets import (TEST_GROUP,
                                           set_download_dir)

from pathlib import PurePath, Path


def cugraph_call(G, min_weight, ensemble_size):
    df = cugraph.ecg(G, min_weight, ensemble_size)
    num_parts = df["partition"].max() + 1
    score = cugraph.analyzeClustering_modularity(
        G, num_parts, df, "vertex", "partition"
    )

    return score, num_parts


def golden_call(name):
    if name == "dolphins":
        return 0.4962422251701355
    if name == "karate-disjoint":
        return 0.38428664207458496
    if name == "netscience":
        return 0.9279554486274719

MIN_WEIGHTS = [0.05, 0.10, 0.15]

ENSEMBLE_SIZES = [16, 32]

set_download_dir(Path(__file__).parents[4] / "datasets")


@pytest.mark.parametrize("dataset", TEST_GROUP)
@pytest.mark.parametrize("min_weight", MIN_WEIGHTS)
@pytest.mark.parametrize("ensemble_size", ENSEMBLE_SIZES)
def test_ecg_clustering(dataset, min_weight, ensemble_size):
    gc.collect()

    # Read in the graph and get a cugraph object
    G = dataset.get_graph(default_direction=False)

    # Get the modularity score for partitioning versus random assignment
    cu_score, num_parts = cugraph_call(G, min_weight, ensemble_size)
    golden_score = golden_call(dataset.metadata['name'])

    # Assert that the partitioning has better modularity than the random
    # assignment
    assert cu_score > (0.95 * golden_score)


@pytest.mark.parametrize("dataset", TEST_GROUP)
@pytest.mark.parametrize("min_weight", MIN_WEIGHTS)
@pytest.mark.parametrize("ensemble_size", ENSEMBLE_SIZES)
def test_ecg_clustering_nx(dataset, min_weight, ensemble_size):
    gc.collect()

    # Read in the graph and get a NetworkX graph
    M = dataset.get_edgelist().rename(
        columns={"src": "0","dst": "1", "wgt": "weight"}
    ).to_pandas()
    G = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.Graph()
    )

    # Get the modularity score for partitioning versus random assignment
    df_dict = cugraph.ecg(G, min_weight, ensemble_size, "weight")

    assert isinstance(df_dict, dict)
