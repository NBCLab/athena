# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 10:49:45 2016

@author: salo
"""

import pandas as pd
from cognitiveatlas.api import get_concept
from cognitiveatlas.api import get_task
#from cognitiveatlas.api import get_disorder  # we won't use disorders until MS[?]


class RelException(Exception):
    def __init__(self, direct, rel_type, term_type):
        Exception.__init__(self,
                           "Unknown relationship direction {0} for relationship type {1} for term type {2}".format(direct,
                                                                                                                   rel_type,
                                                                                                                   term_type))



def clean_id_sheet(df):
    id_df = df[["name", "id"]]
    for i in range(len(id_df)):
        # Protect case of acronyms. Lower everything else.
        print id_df["name"].loc[i]
        try:
            name = str(id_df["name"].loc[i])
        except:
            return id_df["name"].loc[i]
        if name != name.upper():
            id_df["name"][i] = name.lower()

    alias_df = df[["alias", "id"]].astype(str)
    alias_df = alias_df[alias_df["alias"] != ""].reset_index()
    alias2_df = pd.DataFrame(columns=["name", "id"])
    row_counter = 0
    for i in range(len(alias_df)):
        aliases = alias_df["alias"][i].replace("; ", ", ").replace("(", "").replace(")", "").split(", ")
        for alias in aliases:
            # Protect case of acronyms. Lower everything else.
            if alias != alias.upper():
                alias = alias.lower()
            alias2_df.loc[row_counter] = [alias, alias_df["id"][i]]
            row_counter += 1
    id_df = pd.concat([id_df, alias2_df], ignore_index=True)
    return id_df


def create_id_sheet():
    """
    Create spreadsheet with all terms (including synonyms) and their IDs.
    """
    concepts = get_concept().pandas
    tasks = get_task().pandas
    #c_id_df = clean_id_sheet(concepts)
    id_df = clean_id_sheet(tasks)
    return id_df
    #id_df = pd.concat([c_id_df, t_id_df], ignore_index=True)
    
    # Sort by name length (current substitute for searching by term level)
    lens = id_df["name"].str.len()
    lens.sort_values(ascending=False, inplace=True)
    lens = lens.reset_index()
    df = id_df.loc[lens["index"]]
    df = df.reset_index()
    return df


def clean_rel_sheet(df):
    rel_df = pd.DataFrame(columns=["id", "rel_output", "rel_type"])
    row_counter = 0
    for i in range(len(df)):
        id_ = str(df["id"].loc[i])
        
        # First relationship: Category
        category = str(df["id_concept_class"].loc[i])
        if category:
            rel_df.loc[row_counter] = [id_, category, "inCategory"]
            row_counter += 1
        
        # Next, regular relationships
        relationship_list = df["relationships"].loc[i]
        if type(relationship_list) == list:
            for rel in relationship_list:
                if df["type"].loc[i] == "concept":
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
                elif df["type"].loc[i] == "task":
                    """
                    Progenitor of/descendant of and related concepts are not
                    available ATM.
                    """
                    pass
                elif df["type"].loc[i] == "Text":
                    """
                    Disorder structure is strange and disorders are of limited
                    value ATM.
                    """
                    pass
                rel_df.loc[row_counter] = [id_, str(rel["id"]), rel_type]
                row_counter += 1
    return rel_df


def create_rel_sheet():
    concepts = get_concept().pandas
    #tasks = get_task().pandas
    rel_df = clean_rel_sheet(concepts)
    return rel_df


id_df = create_id_sheet()
rel_df = create_rel_sheet()

id_df.to_csv("cogat_ids.csv")
rel_df.to_csv("cog_relationships.csv")
# Disorders have very different structure for synonyms, for some reason.
# disorders = get_disorder().pandas
