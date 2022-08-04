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

import pytest

import cugraph
from cugraph.experimental.datasets import (karate, dolphins,
                                           set_download_dir)
from pathlib import Path

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx


print("Networkx version : {} ".format(nx.__version__))

set_download_dir(Path(__file__).parents[4] / "datasets")
TEST_GROUP = [karate, dolphins]


def calc_k_cores(dataset, directed=True):
    # directed is used to create either a Graph or DiGraph so the returned
    # cugraph can be compared to nx graph of same type.
    NM = dataset.get_edgelist().drop(columns=['wgt']).to_pandas()
    if directed:
        Gnx = nx.from_pandas_edgelist(
            NM, source="src", target="dst", create_using=nx.DiGraph()
        )
    else:
        Gnx = nx.from_pandas_edgelist(
            NM, source="src", target="dst", create_using=nx.Graph()
        )
    G = dataset.get_graph(create_using=cugraph.Graph(directed=directed),
                          ignore_weights=True)

    ck = cugraph.k_core(G)
    nk = nx.k_core(Gnx)
    return ck, nk


def compare_edges(cg, nxg):
    edgelist_df = cg.view_edge_list()
    src, dest = edgelist_df["src"], edgelist_df["dst"]
    assert cg.edgelist.weights is False
    assert len(src) == nxg.size()
    for i in range(len(src)):
        assert nxg.has_edge(src[i], dest[i])
    return True


@pytest.mark.parametrize("dataset", TEST_GROUP)
def test_k_core_Graph(dataset):
    gc.collect()

    cu_kcore, nx_kcore = calc_k_cores(dataset, False)

    assert compare_edges(cu_kcore, nx_kcore)


@pytest.mark.parametrize("dataset", TEST_GROUP)
def test_k_core_Graph_nx(dataset):
    gc.collect()

    NM = dataset.get_edgelist().to_pandas()
    Gnx = nx.from_pandas_edgelist(
        NM, source="src", target="dst", create_using=nx.Graph()
    )
    nc = nx.k_core(Gnx)
    cc = cugraph.k_core(Gnx)

    assert nx.is_isomorphic(nc, cc)


@pytest.mark.parametrize("dataset", TEST_GROUP)
def test_k_core_corenumber_multicolumn(dataset):
    gc.collect()

    cu_M = dataset.get_edgelist()
    cu_M.rename(columns={'src': 'src_0', 'dst': 'dst_0'}, inplace=True)
    cu_M['src_1'] = cu_M['src_0'] + 1000
    cu_M['dst_1'] = cu_M['dst_0'] + 1000

    G1 = cugraph.Graph()
    G1.from_cudf_edgelist(cu_M, source=["src_0", "src_1"],
                          destination=["dst_0", "dst_1"])

    corenumber_G1 = cugraph.core_number(G1)
    corenumber_G1.rename(columns={'core_number': 'values'}, inplace=True)
    corenumber_G1 = corenumber_G1[['0_vertex', '1_vertex', 'values']]

    ck_res = cugraph.k_core(G1, core_number=corenumber_G1)
    G2 = cugraph.Graph()
    G2.from_cudf_edgelist(cu_M, source="src_0",
                          destination="dst_0")
    ck_exp = cugraph.k_core(G2)

    # FIXME: Replace with multi-column view_edge_list()
    edgelist_df = ck_res.edgelist.edgelist_df
    edgelist_df_res = ck_res.unrenumber(edgelist_df, "src")
    edgelist_df_res = ck_res.unrenumber(edgelist_df_res, "dst")
    for i in range(len(edgelist_df_res)):
        assert ck_exp.has_edge(edgelist_df_res["0_src"].iloc[i],
                               edgelist_df_res["0_dst"].iloc[i])
