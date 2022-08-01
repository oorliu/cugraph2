# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
import random

import pytest
import networkx as nx
import pandas as pd
import cudf
import cugraph
from cugraph.experimental.datasets import TEST_GROUP, set_download_dir
from pathlib import Path


set_download_dir(Path(__file__).parents[4] / "datasets")

def cugraph_call(G, partitions):
    df = cugraph.spectralBalancedCutClustering(
        G, partitions, num_eigen_vects=partitions
    )

    score = cugraph.analyzeClustering_edge_cut(
        G, partitions, df, 'vertex', 'cluster'
    )
    return set(df["vertex"].to_numpy()), score


def random_call(G, partitions):
    random.seed(0)
    num_verts = G.number_of_vertices()

    score = 0.0
    for repeat in range(20):
        assignment = []
        for i in range(num_verts):
            assignment.append(random.randint(0, partitions - 1))

        assignment_cu = cudf.DataFrame(assignment, columns=['cluster'])
        assignment_cu['vertex'] = assignment_cu.index

        score += cugraph.analyzeClustering_edge_cut(
            G, partitions, assignment_cu
        )

    return set(range(num_verts)), (score / 10.0)


PARTITIONS = [2, 4, 8]


# Test all combinations of default/managed and pooled/non-pooled allocation


@pytest.mark.parametrize("dataset", TEST_GROUP)
@pytest.mark.parametrize("partitions", PARTITIONS)
def test_edge_cut_clustering(partitions, dataset):
    gc.collect()

    G_edge = dataset.get_graph(default_direction=False)

    # Get the edge_cut score for partitioning versus random assignment
    cu_vid, cu_score = cugraph_call(G_edge, partitions)
    rand_vid, rand_score = random_call(G_edge, partitions)

    # Assert that the partitioning has better edge_cut than the random
    # assignment
    print('graph_file = ', dataset.metadata['name'], ', partitions = ', partitions)
    print(cu_score, rand_score)
    assert cu_score < rand_score


@pytest.mark.parametrize("dataset", TEST_GROUP)
@pytest.mark.parametrize("partitions", PARTITIONS)
def test_edge_cut_clustering_with_edgevals(dataset, partitions):
    gc.collect()

    # Read in the graph and get a cugraph object
    G_edge = dataset.get_graph(default_direction=False)

    # Get the edge_cut score for partitioning versus random assignment
    cu_vid, cu_score = cugraph_call(G_edge, partitions)
    rand_vid, rand_score = random_call(G_edge, partitions)

    # Assert that the partitioning has better edge_cut than the random
    # assignment
    print(cu_score, rand_score)
    assert cu_score < rand_score


# Test to ensure DiGraph objs are not accepted
# Test all combinations of default/managed and pooled/non-pooled allocation


def test_digraph_rejected():
    gc.collect()

    df = cudf.DataFrame()
    df["src"] = cudf.Series(range(10))
    df["dst"] = cudf.Series(range(10))
    df["val"] = cudf.Series(range(10))

    with pytest.deprecated_call():
        G = cugraph.DiGraph()
    G.from_cudf_edgelist(
        df, source="src", destination="dst", edge_attr="val", renumber=False
    )

    with pytest.raises(Exception):
        cugraph_call(G, 2)


@pytest.mark.parametrize("dataset", TEST_GROUP)
@pytest.mark.parametrize("partitions", PARTITIONS)
def test_edge_cut_clustering_with_edgevals_nx(dataset, partitions):
    gc.collect()

    # Read in the graph and create a NetworkX Graph
    # FIXME: replace with utils.generate_nx_graph_from_file()
    NM = dataset.get_edgelist().rename(columns={"src": "0",
                                                "dst": "1",
                                                "wgt": "weight"})
    NM = NM.to_pandas()
    G = nx.from_pandas_edgelist(
                NM, create_using=nx.Graph(), source="0", target="1",
                edge_attr="weight"
    )

    # Get the edge_cut score for partitioning versus random assignment
    df = cugraph.spectralBalancedCutClustering(
        G, partitions, num_eigen_vects=partitions
    )

    pdf = pd.DataFrame.from_dict(df, orient='index').reset_index()
    pdf.columns = ["vertex", "cluster"]
    gdf = cudf.from_pandas(pdf)

    cu_score = cugraph.analyzeClustering_edge_cut(
        G, partitions, gdf, 'vertex', 'cluster'
    )

    df = set(gdf["vertex"].to_numpy())

    Gcu = dataset.get_graph(default_direction=False)
    rand_vid, rand_score = random_call(Gcu, partitions)

    # Assert that the partitioning has better edge_cut than the random
    # assignment
    print(cu_score, rand_score)
    assert cu_score < rand_score
