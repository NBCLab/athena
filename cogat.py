# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 12:35:54 2016

@author: salo
"""

import pandas as pd
import re
from cognitiveatlas.api import get_concept
from cognitiveatlas.api import get_task
#from cognitiveatlas.api import get_disorder  # we won't use disorders until MS[?]


class RelException(Exception):
    def __init__(self, direct, rel_type, term_type):
        Exception.__init__(self, """Unknown relationship direction {0} for 
                                    relationship type {1} for term type {2}""".format(direct, rel_type, term_type))


def clean_id_sheet(df):
    """
    """
    input_df = df[["name", "id", "alias"]]
    id_df = pd.DataFrame(columns=["term", "id", "preferred term"])
    
    row_counter = 0
    for i in range(len(input_df)):
        # Protect case of acronyms. Lower everything else.
        # Also convert apostrophe symbols to apostrophes.
        name = input_df["name"].loc[i].encode("utf-8").strip()
        name = str(name.replace("&#39;", "'"))

        if name:
            if name != name.upper():
                name = name.lower()
            id_df.loc[row_counter] = [name, input_df["id"].loc[i], name]
            row_counter += 1
        
        aliases = input_df["alias"].loc[i].encode("utf-8").strip()
        aliases = re.sub(r"\s+\(([^)]+)\)", "", aliases)
        aliases.replace("; ", ", ").split(", ")
        aliases = [alias for alias in aliases if alias]
        for alias in aliases:
            # Protect case of acronyms. Lower everything else.
            # Also convert apostrophe symbols to apostrophes.
            alias = str(alias.replace("&#39;", "'"))
        
            if alias != alias.upper():
                alias = alias.lower()
            id_df.loc[row_counter] = [alias, input_df["id"].loc[i], name]
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
    rel_df = pd.DataFrame(columns=["id", "rel_output", "rel_type"])
    row_counter = 0
    for i in range(len(df)):
        id_ = str(df["id"].loc[i])
        
        # First relationship: Category
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
                            raise RelException(rel["direction"], rel["relationship"], df["type"].loc[i])
                    elif rel["relationship"] == "part of":
                        if rel["direction"] == "parent":
                            rel_type = "isPartOf"
                        elif rel["direction"] == "child":
                            rel_type = "hasPart"
                        else:
                            raise RelException(rel["direction"], rel["relationship"], df["type"].loc[i])
                    else:
                        raise Exception("Unknown relationship {0} for type {1}.".format(rel["relationship"], df["type"].loc[i]))
                    rel_df.loc[row_counter] = [id_, str(rel["id"]), rel_type]
                    row_counter += 1

        elif df["type"].loc[i] == "task":
            task = get_task(id=id_, silent=True).json
            if "concepts" in task[0].keys():
                for concept in task[0]["concepts"]:
                    rel_df.loc[row_counter] = [id_, concept["concept_id"], "relatedConcept"]
                    row_counter += 1
            
            if "disorders" in task[0].keys():
                for disorder in task[0]["disorders"]:
                    rel_df.loc[row_counter] = [id_, disorder["id_disorder"], "relatedDisorder"]
                    row_counter += 1
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

id_df = create_id_sheet()
rel_df = create_rel_sheet()

id_df.to_csv("cogat_ids.csv", index=False)
rel_df.to_csv("cog_relationships.csv", index=False)
