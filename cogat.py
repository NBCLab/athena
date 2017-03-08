# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 12:35:54 2016

@author: salo
"""

from __future__ import division
from os.path import join, basename, splitext
import pandas as pd
import re
import numpy as np
from glob import glob
import itertools
from cognitiveatlas.api import get_concept
from cognitiveatlas.api import get_task
from cognitiveatlas.api import get_disorder
from cogat_weighting_schemes import get_weights

# Constants
spell_file = join("/home/data/nbc/athena/athena-data/misc/english_spellings.csv")
spell_df = pd.read_csv(spell_file, index_col="UK")
spell_dict = spell_df["US"].to_dict()


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


def clean_string(string):
    """
    Clean CogAt terms.
    """
    # Convert apostrophe symbols to apostrophes. Also strip whitespace.
    string = str(string.replace("&#39;", "'")).strip()
    
    # Convert British to American English
    pattern = re.compile(r"\b(" + "|".join(spell_dict.keys()) + r")\b")
    string = pattern.sub(lambda x: spell_dict[x.group()], string)

    # Terms that start and end with parentheses are generally abbreviation aliases.
    # Also remove unpaired parentheses
    if string.startswith("(") and string.endswith(")"):
        string = string.replace("(", "").replace(")", "")
    elif "(" in string and ")" not in string:
        string = string.replace("(", "")
    elif "(" not in string and ")" in string:
        string = string.replace(")", "")
    
    # literalize plus signs
    string = string.replace("+", "\+")
    
    # Create list for alternate forms
    string_set = [string]

    # For one alternate form, put contents of parentheses at beginning of term
    if "(" in string:
        prefix = string[string.find("(")+1:string.find(")")]
    else:
        prefix = ""
    
    # Remove parenthetical statements.
    string = re.sub(r"\([^\)]*\)", "", string)
    string = re.sub(r"\[[^\]]*\]", "", string)  
    string_set.append(string)

    if prefix:
        string = "{0} {1}".format(prefix, string)
        string_set.append(string)

    # Remove extra spaces.
    string_set = [re.sub("\s+", " ", s).lower() for s in string_set]
    string_set = [s.strip() for s in string_set]
    
    new_strings = []
    for s in string_set:
        new_strings.append(s.replace("-", ""))
        new_strings.append(s.replace("-", " "))
        new_strings.append(s.replace("'s", " s"))
        new_strings.append(s.replace("'s", "s"))
        new_strings.append(s.replace("-", "").replace("'s", " s"))
        new_strings.append(s.replace("-", "").replace("'s", "s"))
        new_strings.append(s.replace("-", " ").replace("'s", " s"))
        new_strings.append(s.replace("-", " ").replace("'s", "s"))
        new_strings.append(s.replace(" / ", "/"))
    string_set += new_strings
    
    # Remove duplicates
    string_set = list(set(string_set))
    return string_set


def clean_disorders_id_sheet(df):
    """
    """
    id_df = pd.DataFrame(columns=["term", "id", "preferred term"])
    
    row_counter = 0
    for i in range(len(df)):
        name = df["name"].loc[i].encode("utf-8").strip()
        names = clean_string(name)
        if names:
            pref_name = names[0]
            for name in names:
                id_df.loc[row_counter] = [name, df["id"].loc[i], pref_name]
                row_counter += 1
            
            aliases = [alias_dict["synonym"] for alias_dict in df["synonyms"].loc[i]]
            for j in range(len(aliases)):
                aliases[j] = clean_string(aliases[j])
    
            # Add aliases to DataFrame
            alias_list = [item for sublist in aliases for item in sublist]
            alias_list = list(set(alias_list))
            if alias_list:
                for alias in alias_list:
                    id_df.loc[row_counter] = [alias, df["id"].loc[i], pref_name]
                    row_counter += 1
    return id_df


def clean_id_sheet(df):
    """
    """
    id_df = pd.DataFrame(columns=["term", "id", "preferred term"])
    
    row_counter = 0
    for i in range(len(df)):
        name = df["name"].loc[i].encode("utf-8").strip()
        names = clean_string(name)
        if names:
            pref_name = names[0]
            for name in names:
                id_df.loc[row_counter] = [name, df["id"].loc[i], pref_name]
                row_counter += 1
        
            aliases = df["alias"].loc[i].encode("utf-8").strip()
            aliases = aliases.replace("; ", ", ").split(",")
            aliases = [alias for alias in aliases if alias]
            for j in range(len(aliases)):
                aliases[j] = clean_string(aliases[j])

            # Add aliases to DataFrame
            alias_list = [item for sublist in aliases for item in sublist]
            alias_list = list(set(alias_list))
            if alias_list:
                for alias in alias_list:
                    id_df.loc[row_counter] = [alias, df["id"].loc[i], pref_name]
                    row_counter += 1
    return id_df


def create_id_sheet():
    """
    Create spreadsheet with all terms (including synonyms) and their IDs.
    """
    concepts = get_concept(silent=True).pandas
    tasks = get_task(silent=True).pandas
    disorders = get_disorder(silent=True).pandas
    c_id_df = clean_id_sheet(concepts)
    t_id_df = clean_id_sheet(tasks)
    d_id_df = clean_disorders_id_sheet(disorders)
    id_df = pd.concat([c_id_df, t_id_df, d_id_df], ignore_index=True)
    
    # Sort by name length (current substitute for searching by term level)   
    lens = id_df["term"].str.len()
    lens.sort_values(ascending=False, inplace=True)
    lens = lens.reset_index()
    df = id_df.loc[lens["index"]]
    
    # Keep the first of duplicates (not the strongest)
    df.drop_duplicates(subset=["term"], inplace=True)
    df = df.reset_index(drop=True)
    df = df.replace("", np.nan)
    df.dropna(subset=["term"], inplace=True)

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
        if "id_concept_class" in df.columns:
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
        # Disorder
        elif df["type"].loc[i] == "Text":
            disorder = get_disorder(id=id_, silent=True).json
            if "is_a" in disorder[0].keys():
                if str(disorder[0]["is_a"]):
                    rel_df.loc[row_counter] = [id_, str(disorder[0]["is_a"]), "childOf"]
                    row_counter += 1
    # Keep the first of duplicates (not the strongest)
    rel_df.drop_duplicates(subset=["input", "output"], inplace=True)
    return rel_df


def create_rel_sheet(id_df):
    """
    """
    concepts = get_concept().pandas
    tasks = get_task().pandas
    disorders = get_disorder().pandas
    c_rel_df = clean_rel_sheet(concepts)
    t_rel_df = clean_rel_sheet(tasks)
    d_rel_df = clean_rel_sheet(disorders)
    rel_df = pd.concat([c_rel_df, t_rel_df, d_rel_df], ignore_index=True)

    cogat_terms = id_df["id"].tolist()
    rel_inputs = rel_df["input"].tolist()
    rel_outputs = rel_df["output"].tolist()
    rel_terms = list(set(rel_inputs + rel_outputs))
    
    # Some terms may no longer exist, but may still be referenced in assertions.
    # These terms must be removed from the set of relationships.
    not_in_gaz = list(set(rel_terms) - set(cogat_terms))
    drop = [term for term in not_in_gaz if not term.startswith("ctp")]
    
    keep_terms = sorted(list(set(rel_terms) - set(drop)))
    
    rel_df = rel_df.loc[rel_df["input"].isin(keep_terms)]
    rel_df = rel_df.loc[rel_df["output"].isin(keep_terms)]
    
    return rel_df


def weight_rels(rel_df, weighting_scheme="none"):
    """
    Based on a relationship legend (e.g., term1 is a kindOf term2) and a
    weighting scheme (e.g., if a term is found, you also count toward all terms
    that term is a kindOf), create a weight matrix.
    
    Imagine term1 is a kindOf term2, and term2 is a kindOf term3. If you find
    term1 in the text, you need to count toward term2 AND term3. The goal here
    is to make sure the row corresponding to term1 in the weight matrix has the
    correct weights for term2 and term3, without getting caught in any infinite
    loops.
    NOTE: If a term is connected to another term through more than one path
    (which may have different weights) we use the max of the weights.
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
    weight_df.index.name = "term"
    del weight_df.columns.name
    
    # Get weights for all term-term relationships, regardless of path length.
    raw_weights = np.eye(weight_df.shape[0])
    
    # Skip terms with no relationships
    row_idx = np.where(weight_df.sum(axis=1).values>0)[0]
    for i_row in row_idx:
        arr = np.zeros((1, weight_df.shape[0]))
        arr[0, i_row] = 1
        weights = np.copy(arr)
        while np.any(arr):
            # Get next order of relationships for term
            arr = np.dot(arr, weight_df.values)
            
            # If no weights increase, the weights are stable and we can quit
            temp_arr = np.maximum(weights, arr)
            if np.array_equal(weights, temp_arr):
                break
            else:
                weights = temp_arr

        raw_weights[i_row, :] = weights
    
    weight_df[weight_df.columns] = raw_weights

    return weight_df


def create_re(term):
    words = term.split(" ")
    regex = "\\s*(\\(.*\\))?\\s*".join(words)
    regex = "\\b"+regex+"\\b"
    pattern = re.compile(regex, re.DOTALL)
    return pattern


def extract_cogat(cogat_df, text_dir, out_dir):
    """
    Creates feature table for Cognitive Atlas terms from full, unstemmed text.
    Just a first pass.
    """
    # Read in features
    pmids = sorted([basename(splitext(f)[0]) for f in glob(join(text_dir, '*.txt'))])
    gazetteer = sorted(cogat_df["id"].unique().tolist())

    # Create regex dictionary
    regex_dict = {}
    for term in cogat_df['term'].values:
        regex_dict[term] = create_re(term)
    
    # Count
    count_array = np.zeros((len(pmids), len(gazetteer)))
    for i, pmid in enumerate(pmids):
        with open(join(text_dir, pmid+'.txt'), 'r') as fo:
            text = fo.read().lower()
        
        for row in cogat_df.index:
            term = cogat_df["term"].loc[row]
            term_id = cogat_df["id"].loc[row]
            
            col_idx = gazetteer.index(term_id)
            
            pattern = regex_dict[term]
            count = len(re.findall(pattern, text))
            count_array[i, col_idx] += count
            text = re.sub(pattern, term_id, text)

        with open(join(out_dir, pmid+'.txt'), 'w') as fo:
            fo.write(text)
        
    # Create and save output
    count_df = pd.DataFrame(columns=gazetteer, index=pmids, data=count_array)
    count_df.index.name = 'pmid'
    return count_df


def apply_weights(input_df, weight_df):
    """
    Apply weights once.
    """
    weight_df = weight_df.reindex_axis(sorted(weight_df.columns), axis=1).sort()
    input_df = input_df.reindex_axis(sorted(input_df.columns), axis=1)
    
    if not (set(weight_df.columns) == set(input_df.columns)):
        raise Exception("Columns do not match between DataFrames!")

    weighted_df = input_df.dot(weight_df)
    return weighted_df


def run(data_dir='/home/data/nbc/athena/athena-data2/', sources=['abstract', 'full']):
    #id_df = create_id_sheet()
    #id_df.to_csv(join(data_dir, 'gazetteers/cogat_ids.csv'))
    id_df = pd.read_csv(join(data_dir, 'gazetteers/cogat_ids.csv'))
    #rel_df = create_rel_sheet(id_df)
    #rel_df.to_csv(join(data_dir, 'gazetteers/cogat_rels.csv'))
    #rel_df = pd.read_csv(join(data_dir, 'gazetteers/cogat_rels.csv'))
    #weight_df = weight_rels(rel_df, 'ws2_up')
    weight_df = pd.read_csv(join(data_dir, 'gazetteers/cogat_weights.csv'),
                            index_col='term')

    for i, source in enumerate(sources):
        text_folder = join(data_dir, 'text/cleaned_{0}/'.format(source))
        out_dir = join(data_dir, 'text/cogat_cleaned_{0}/'.format(source))
        count_df = extract_cogat(id_df, text_folder, out_dir)
        
        if i == 0:
            # Reduce weight_df and count_df by their common CogAt terms.
            cogat_input = count_df.columns.values.tolist()
            cogat_weight = weight_df.columns.values.tolist()
            
            not_in_input = list(set(cogat_weight) - set(cogat_input))
            
            add = [term for term in not_in_input if term.startswith("ctp")]
            drop = [term for term in not_in_input if not term.startswith("ctp")]
            
            new_weight = sorted(list(set(cogat_weight) - set(drop)))
            
            weight_df = weight_df[new_weight]
            weight_df = weight_df[weight_df.index.isin(new_weight)]
            weight_df.to_csv(join(data_dir, 'gazetteers/cogat_weights_adjusted.csv'))            
        
        # Add to input
        for val in add:
            count_df[val] = 0
        
        count_df = count_df[new_weight]
        count_df.to_csv(join(data_dir, 'features/cogat_counts_{0}.csv'.format(source)))
     
        weighted_df = apply_weights(count_df, weight_df)
        weighted_df.to_csv(join(data_dir, 'features/cogat_{0}.csv'.format(source)))
