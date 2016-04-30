"""
Run v1.1 from beginning to end.
"""
import process_data
import os
import gazetteers
import feature_extraction
import pandas as pd
import classifier_handler

data_dir = "/home/data/nbc/athena/v1.1-data/"

process_data.stem_corpus(data_dir)
process_data.label_data(data_dir)
for text_type in ["full", "combined"]:
    # Process data
    type_dir = os.path.join(data_dir, text_type)
    label_file = os.path.join(type_dir, "labels/full.csv")
    process_data.split_data(label_file, test_percent=0.33)
    
    # Generate gazetteer
    gaz_dir = os.path.join(type_dir, "gazetteers/")
    stem_text_dir = os.path.join(data_dir, "text", "stemmed_"+text_type)

    df = pd.read_csv(label_file)
    pmids = df["pmid"].astype(str).tolist()

    nbow_gaz = gazetteers.generate_nbow_gazetteer(pmids, stem_text_dir)
    print("Completed nbow gaz.")

    # Save gazetteer
    gazetteers.save_gaz(nbow_gaz, gaz_dir, "nbow")

feature_extraction.extract_features()
import summarize_datasets
classifier_handler.run_classifiers(data_dir)
