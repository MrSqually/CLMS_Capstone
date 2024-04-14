#!/usr/bin/env python3
# CLMS Capstone | Dean Cahill 
# Clustering Preprocess Code

# Uses the SciKit-Learn implementation of Affinity Propogation 
# to generate clusters 

# ============================================================|
# Import Statements
import argparse 
from tqdm import tqdm
from typing import Union, Generator
import ijson
import logging
import os 
import toml

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import torch


#LOG
logger = logging.getLogger(__name__)

# ============================================================|
# Clustering Loop
# TODO run cluster algorithm
def run_cluster_algorithm(docs: torch.Tensor, 
                          **params):
    """"""
    max_sil_score: int = -1 
    for i in tqdm(range(params["num_epochs"])):
        affinity_prop = AffinityPropagation(docs, **params["af_params"]).fit(docs)
        
        results = {"#_clusters":len(affinity_prop.cluster_centers_indices_),
                    "exemplars":affinity_prop.cluster_centers_indices_,
                    "labels":affinity_prop.labels_,
                    "silhouette_score": metrics.silhouette_score(docs, affinity_prop.labels_, metric="sqeuclidean")}
        
        if results["silhouette_score"] > max_sil_score:
            print("New highest silhouette score")
            max_sil_score == results["silhouette_score"]
            if results["#_clusters"] > params["min_clusters"]:
                save_clusters(affinity_prop, docs, results)
        display_result(docs, results, write=True)
            
    

#TEST save clusters
def save_clusters(docs: torch.Tensor,
                  results:dict):
    """Serialize Clusters into a set of pt files,
    to enable their use as sets"""
    for k in range(results["num_clusters"]):
        class_members = results["labels"] == k
        class_docs = docs[class_members]
        torch.save(class_docs, f"{results['silhouette_score']}_cluster_{k}.pt")


#TEST display result
def display_result(docs, results, write=None):
    """Constructs visualization of clusters"""
    import matplotlib.pyplot as plt 
    plt.close("all")
    plt.figure(1)
    plt.clf()

    colors = plt.cycler("color", plt.cm.viridis(torch.linspace(0,1,4)))

    for k, col in zip(range(results["#_clusters"], colors)):
        class_members = results["labels"] == k
        cluster_center = docs[results["exemplars"][k]]
        plt.scatter(
            docs[class_members, 0], docs[class_members, 1], color=col["color"], marker="."
        )   
        plt.scatter(
            cluster_center[0], cluster_center[1], s=14, color=col["color"], marker="o"
        )
        for x in docs[class_members]:
            plt.plot(
                [cluster_center[0], x[0]], [cluster_center[1], x[1]], color = col["color"]
            )
    plt.title(f"Estimated number of clusters: {results['#_clusters']}")
    if write:
        plt.savefig(write)
    plt.show()

# ============================================================|
# Optimal Parameter Search 
# ============================================================|
# Main
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clusters from dataset")
    parser.add_argument("-c", "--config",
                        help="",
                        default="config/config.toml")

    return parser.parse_args()

def get_config(config:str | os.PathLike) -> dict:
    with open(config) as f:
        config_dict = toml.load(f)
    return config_dict

def main(args: argparse.Namespace):
    # TODO main
    parameters = get_config(args.config)
    print(parameters.keys())

if __name__ == "__main__":
    args = parse_args()
    main(args)