# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 12:35:54 2016

@author: salo
"""

from __future__ import division
import pandas as pd
import re
from cognitiveatlas.api import get_concept
from cognitiveatlas.api import get_task
#from cognitiveatlas.api import get_disorder  # we won't use disorders until MS[?]
import numpy as np
import itertools
from cogat_weighting_schemes import get_weights
import time


class RelException(Exception):
    def __init__(self, term_type, rel_type=None, direct=None):
        
        if direct != None and rel_type != None:
            Exception.__init__(self, """Unknown relationship direction {0} for 
                                        relationship type {1} for term type 
                                        {2}""".format(direct, rel_type,
                                                      term_type))
        elif rel_type != None:
            Exception.__init__(self, """Unknown relationship type {0} for term 
                                        type {1}""".format(rel_type,
                                                           term_type))
        else:
            Exception.__init__(self, """Unknown term type {0}""".format(term_type))


def clean_id_sheet(df):
    """
    """
    id_df = pd.DataFrame(columns=["term", "id", "preferred term"])
    
    row_counter = 0
    for i in range(len(df)):
        # Protect case of acronyms. Lower everything else.
        # Also convert apostrophe symbols to apostrophes.
        name = df["name"].loc[i].encode("utf-8").strip()
        name = str(name.replace("&#39;", "'"))

        if name:
            if name != name.upper():
                name = name.lower()
            id_df.loc[row_counter] = [name, df["id"].loc[i], name]
            row_counter += 1
        
        aliases = df["alias"].loc[i].encode("utf-8").strip()
        aliases = re.sub(r"\s+\(([^)]+)\)", "", aliases)
        aliases.replace("; ", ", ").split(", ")
        aliases = [alias for alias in aliases if alias]
        for alias in aliases:
            # Protect case of acronyms. Lower everything else.
            # Also convert apostrophe symbols to apostrophes.
            alias = str(alias.replace("&#39;", "'"))
        
            if alias != alias.upper():
                alias = alias.lower()
            id_df.loc[row_counter] = [alias, df["id"].loc[i], name]
            row_counter += 1
    return id_df


def create_id_sheet():
    """
    Create spreadsheet with all terms (including synonyms) and their IDs.
    """
    concepts = get_concept(silent=True).pandas
    tasks = get_task(silent=True).pandas
    c_id_df = clean_id_sheet(concepts)
    t_id_df = clean_id_sheet(tasks)
    id_df = pd.concat([c_id_df, t_id_df], ignore_index=True)
    
    # Sort by name length (current substitute for searching by term level)
    lens = id_df["term"].str.len()
    lens.sort_values(ascending=False, inplace=True)
    lens = lens.reset_index()
    df = id_df.loc[lens["index"]]
    df = df.reset_index(drop=True)
    return df


def clean_rel_sheet(df):
    """
    """
    rel_df = pd.DataFrame(columns=["input", "output", "rel_type"])
    row_counter = 0
    for i in range(len(df)):
        id_ = str(df["id"].loc[i])
        
        # First relationship: Self
        rel_df.loc[row_counter] = [id_, id_, "isSelf"]
        row_counter += 1
        
        # Second relationship: Category
        category = str(df["id_concept_class"].loc[i])
        if category:
            rel_df.loc[row_counter] = [id_, category, "inCategory"]
            row_counter += 1
        
        if df["type"].loc[i] == "concept":
            concept = get_concept(id=id_, silent=True).json
            
            if "relationships" in concept[0].keys():
                relationships = concept[0]["relationships"]
                for rel in relationships:
                    if rel["relationship"] == "kind of":
                        if rel["direction"] == "parent":
                            rel_type = "isKindOf"
                        elif rel["direction"] == "child":
                            rel_type = "hasKind"
                        else:
                            raise RelException(df["type"].loc[i],
                                               rel["relationship"],
                                               rel["direction"])
                    elif rel["relationship"] == "part of":
                        if rel["direction"] == "parent":
                            rel_type = "isPartOf"
                        elif rel["direction"] == "child":
                            rel_type = "hasPart"
                        else:
                            raise RelException(df["type"].loc[i],
                                               rel["relationship"],
                                               rel["direction"])
                    else:
                        raise RelException(df["type"].loc[i],
                                           rel["relationship"])
                    rel_df.loc[row_counter] = [id_, str(rel["id"]), rel_type]
                    row_counter += 1

        elif df["type"].loc[i] == "task":
            task = get_task(id=id_, silent=True).json
            if "concepts" in task[0].keys():
                for concept in task[0]["concepts"]:
                    rel_df.loc[row_counter] = [id_, concept["concept_id"],
                                               "relatedConcept"]
                    row_counter += 1
    # Keep the first of duplicates (not the strongest)
    rel_df.drop_duplicates(subset=["input", "output"], inplace=True)
    return rel_df


def create_rel_sheet():
    """
    """
    concepts = get_concept().pandas
    tasks = get_task().pandas
    c_rel_df = clean_rel_sheet(concepts)
    t_rel_df = clean_rel_sheet(tasks)
    rel_df = pd.concat([c_rel_df, t_rel_df], ignore_index=True)
    return rel_df


def create_weighted_rel_sheet(rel_df, weighting_scheme="none"):
    """
    """
    weights = get_weights(weighting_scheme)
    
    existing_rels = rel_df[["input", "output"]].values.tolist()
    
    term_ids = np.unique(rel_df[["input", "output"]])
    all_possible_rels = [list(pair) for pair in list(itertools.product(term_ids,
                                                                       term_ids))]
    all_possible_rels = set(map(tuple, all_possible_rels))
        
    new_rels = list(all_possible_rels - set(map(tuple, existing_rels)))
    
    weight_df = pd.DataFrame(columns=["input", "output"], data=new_rels)
    weight_df["weight"] = np.zeros(len(weight_df))
    row_counter = len(weight_df)
    
    for rel in existing_rels:
        rel_idx = np.intersect1d(np.where(rel_df["input"]==rel[0])[0],
                                 np.where(rel_df["output"]==rel[1])[0])[0]
        rel_type = rel_df["rel_type"].iloc[rel_idx]
        if rel_type in weights.keys():
            if type(weights[rel_type]) == dict:
                num = weights[rel_type]["num"]
                den = weights[rel_type]["den"]
                if den == "n":
                    den = len(np.intersect1d(np.where(rel_df["input"]==rel[0])[0],
                                             np.where(rel_df["rel_type"]==rel_type)[0]))
                else:
                    raise Exception("Unknown string in denominator: {0}".format(den))
                weight = num / den
            else:
                weight = weights[rel_type]
        else:
            weight = 0
        weight_df.loc[row_counter] = [rel[0], rel[1], weight]
        row_counter += 1

    weight_df = weight_df.pivot(index="input", columns="output", values="weight")
    del weight_df.index.name
    del weight_df.columns.name

    return weight_df


def apply_weights(input_df, weight_df):
    """
    """
    weight_df = weight_df.reindex_axis(sorted(weight_df.columns), axis=1).sort()
    input_df = input_df.reindex_axis(sorted(input_df.columns), axis=1)
    weighted_df = input_df.dot(weight_df)
    return weighted_df

# Create and save ID, relationship, and weight files.
id_df = create_id_sheet()
id_df.to_csv("cogat_ids.csv", index=False)

rel_df = create_rel_sheet()
rel_df.to_csv("cogat_relationships.csv", index=False)

weighting_scheme = "ws2"
weight_df = create_weighted_rel_sheet(rel_df, weighting_scheme)
date = time.strftime('%Y%m%d')  # We have date here because CogAt is always changing.
weight_df.to_csv("cogat_weights_{0}_{1}.csv".format(weighting_scheme, date),
                 index=True)
