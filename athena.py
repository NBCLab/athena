import glob
import os
import numpy as np
import pandas as pd
import pickle
import string
from collections import defaultdict
from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC


class Athena:
    """
    Class containing all abstracts and tests.
    """
    def __init__(self):
        """-----    Text Variables   -----"""
        
        data_folder = os.path.join(os.path.expanduser("~"), "dAthena/data/")
        
        # Location of abstracts' plain-text
        #self.corpus_directory = os.path.join(data_folder, "stemmed/abstracts/*.txt")
        #self.corpus_directory = os.path.join(data_folder, "stemmed/methods/*.txt")
        #self.corpus_directory = os.path.join(data_folder, "stemmed/combined/*.txt")
        
        #self.corpus_directory = os.path.join(data_folder, "abstracts/*.txt")
        #self.corpus_directory = os.path.join(data_folder, "methods/*.txt")
        self.corpus_directory = os.path.join(data_folder, "combined/*.txt")
        #self.corpus_directory = os.path.join(data_folder, "2013_abstracts/*.txt")

        # Location of stopword list
        self.stopword_file = os.path.join(data_folder,"misc_data/onix_stopwords.txt")
        
        # Total # of abstracts/methods loaded
        self.text_files_loaded = 0
        
        # Names of all loaded files
        self.filenames = None
        
        # Holds all of the plain text
        self.text_corpus = None

        """-----    Metadata Variables   -----"""
        # Location of metadata
        self.meta_data_directory = os.path.join(data_folder, "meta_data/*.csv")
        self.meta_data = None
        
        # Column names we want to keep
        # These variables need more informative names
        self.index_name = ["Year", "First Author", "Journal", "PubMed ID"]
        self.column_name = ["Diagnosis", "Stimulus Modality", "Response Modality",
                            "Response Type", "Stimulus Type", "Instructions",
                            "Behavioral Domain", "Paradigm Class"]
        self.column_name2 = ["all", "Diagnosis", "Stimulus Modality",
                             "Response Modality", "Response Type", "Stimulus Type",
                             "Instructions", "Behavioral Domain", "Paradigm Class"]
        self.kept_columns = ["PubMed ID", "Diagnosis", "Stimulus Modality",
                             "Response Modality", "Response Type", "Stimulus Type",
                             "Instructions", "Behavioral Domain", "Paradigm Class"]

        """------ Combined Data ------"""
        self.combined_meta_data = None
        self.combined_df = None

        """------- Training & Test Data ----"""
        self.train_data = None
        self.test_data = None
        self.train_label = None
        self.test_label = None

        """------- MultiLabelBinarizer ----"""
        self.mlb = None
        self.label_mlb = None
        self.label_text_dict = None
        self.label_bin_dict = None
        self.label_bin_df = None
        self.label_df = None
        self.label_dimension_dict = None

        """-----    Pipeline   -----"""
        self.clf_pipeline = None
        self.clfs = None
        self.clf_names = None
        
        self.results = None
        self.text = None

        self.grid_searches = None

        # Label dim stuff
        self.n_feature = None
        self.dimension_end = None
        self.dimension_beg = None
        self.test = None
        self.test2 = None

        """ Test """
        self.bnb_clf = None
        self.tvect = None
        self.x1 = None
        self.test_pred = None
        self.test_f1 = None


    def read_stopwords(self):
        """
        Reads in stopwords from file.
        """
        print("Reading stopwords...")
        with open(self.stopword_file) as fo:
            self.stopwords = fo.read().split()
        return self


    def read_text(self):
        """
        Loads in text.
        Corpus can be accessed with corpus.text_corpus["dictkey"]
        dictkeys are the filenames minus the extension
        e.g. corpus.text_corpus["11467915"] will give you text from 11468915_a_p.txt
        """
        print("Reading text data...")
        temp_corpus = dict()
        for filename in sorted(glob.glob(self.corpus_directory)):
            # Read in text from file
            with open(filename) as fo:
                text = fo.read()

            # The [:-8] gets rid of the last 8 chars of file name
            # 11467915_a_p.txt -> 11467915
            # Abstracts ext_len = -8, methods ext_len = -4
            ext_len = -4
            temp_corpus[os.path.basename(filename)[:ext_len]] = text

        # filenames now contains all the dictionary kegs (filenames)
        self.filenames = sorted(temp_corpus)
        
        # text_corpus contains all the plaintext keyed with filenames
        self.text_corpus = temp_corpus
        
        # update number of files
        self.text_files_loaded = len(self.filenames)
        
        print("Files loaded: {0}".format(self.text_files_loaded))
        return self


    def read_meta_data(self):
        """
        Loads in all metadata from .csv files (pain.csv, face.csv, etc)
        """
        print("Reading metadata...")
        
        # Read in all of the metadata files
        df = [pd.read_csv(i, dtype=np.str) for i in sorted(glob.glob(self.meta_data_directory))]
        
        # Now we have to join all the separate tables stored in df
        df = pd.concat(df, ignore_index=True)
        
        # Keep useful labels we want
        df = df.loc[:, self.index_name + self.column_name]
        
        # Drop duplicates
        df = df.drop_duplicates()
        
        # Drop rows with null PMIDs
        df = df[(df["PubMed ID"]!="null")]

        # Drop rows who are missing columns
        df = df.dropna()
        df["PubMed ID"] = df["PubMed ID"].apply(int)
        
        # Sort the table
        df = df.sort_values(by=["PubMed ID", "Year", "First Author", "Journal"])
        df["PubMed ID"] = df["PubMed ID"].apply(str)
        #print df["PubMed ID"]

        self.meta_data = df
        return df


    def _merge_series(self, series, curr_column):
        """
        Merges experiments with multiple labels in one label dimension.
        """
        label_zip =  zip(series)
        end_set = set()

        for each_val in label_zip:
            temp_val = [v.split("|") for v in each_val]
            for v in temp_val:
                v = [s.strip() for s in v]
                end_set.update(v)
        
        if "None" in end_set:
            end_set.remove("None")
            end_set.add(curr_column + "_None")

        return end_set


    def combine_meta_data(self):
        """
        A lot of the metadata has multiple rows for one PMID so let's merge
        them.
        """
        print("Combining metadata...")
        
        # New dataframe with the index being PMIDs
        df = pd.DataFrame(index=self.meta_data["PubMed ID"].unique(),
                          columns=self.column_name, dtype="string")

        # Loop over rows
        for row_index, current_row in self.meta_data.iterrows():
            # Grab current PMID
            current_pmid = current_row["PubMed ID"]
            
            # Grab all rows which match this PMID (multiple rows are from same paper)
            current_record = self.meta_data[self.meta_data["PubMed ID"]==current_pmid]
            
            # Loop over all columns we want to keep
            for curr_column in self.column_name:
                # Save unique values for each column in each PMID into df
                df.loc[current_pmid, curr_column] = self._merge_series(current_record[curr_column].unique(),
                                                                       curr_column)
        self.combined_meta_data = df
        return self


    def combine_data(self):
        """
        Combines the metadata + abstracts into one table.
        """
        print("Combining abstracts and metadata...")
        
        # Metadata table basis for our new combined data array
        self.combined_df = self.combined_meta_data
        
        # Add an abstract text column
        self.combined_df["Abstract Text"] = ""

        for row_index, current_row in self.combined_meta_data.iterrows():
            try:
                current_pmid = row_index
                current_abs = self.text_corpus[current_pmid]
                self.combined_df.loc[current_pmid, "Abstract Text"] = current_abs
            except Exception as e:
                # Throws key error if we didn't find the abstract text, so drop the column from the table
                print e
                self.combined_df = self.combined_df.drop(row_index)
                pass
        # print self.combined_df
        return self


    def process_text(text):
        """
        Tokenize text and stem words, removing punctuation.
        Adapted from http://tech.swamps.io/recipe-text-clustering-using-nltk-and-scikit-learn/
        """
        text = text.translate(None, string.punctuation)
        tokens = word_tokenize(text)
        stemmer = LancasterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
        return tokens


    def get_unique_labels(self, paradigm_label):
        """
        Grab all unique labels for a paradigm class!
        """
        paradigm_list = []
        for i in self.combined_df[paradigm_label]:
            for c_set in i:
                paradigm_list.append(c_set)
        paradigm_set = set(paradigm_list)
        return paradigm_set


    def split_data(self, partition_size=0.30):
        """
        Splits the data into two partitions (training and testing data).
        """
        print("Partitioning Data...")
        self.train_data, self.test_data, self.train_label, self.test_label = train_test_split(self.combined_df,
                                                                                              self.label_bin_df,
                                                                                              test_size=partition_size)
        return self


    def split_data_abs(self):
        """
        Splits the data into two partitions (training and testing data).
        """
        print("Partitioning Data...")
        self.train_data = self.combined_df.copy()
        self.test_data = self.combined_df.copy()
        self.train_label = self.label_bin_df.copy()
        self.test_label = self.label_bin_df.copy()
        return self


    def binarize(self):
        """
        Sets up the MLB as well as dictionaries corresponding to the binary
        matrix.
        """
        self.label_text_dict = defaultdict(list)

        for index, current_row in self.combined_df.iterrows():
            for current_column_name in self.column_name:
                self.label_text_dict[current_column_name].append(current_row[current_column_name])

        self.label_bin_dict = {key:MultiLabelBinarizer().fit_transform(label_list) for key, label_list in self.label_text_dict.items()}
        self.label_bin_df = pd.DataFrame(np.concatenate([self.label_bin_dict[k] for k in self.column_name], 1))
        self.combined_df.index = range(len(self.combined_df.index))
        self.label_df = pd.concat([self.combined_df, self.label_bin_df], axis=1)

        label_mlb_dict = {key:MultiLabelBinarizer().fit(label_list) for key, label_list in self.label_text_dict.items()}
        label_mlb = MultiLabelBinarizer()
        label_mlb.classes_ = np.concatenate([label_mlb_dict[key].classes_ for key in self.column_name])
        # self.label_mlb = label_mlb
        
        label_dimension_dict = {key:set(label_mlb_dict[key].classes_) for key in self.column_name}
        self.label_dimension_dict = label_dimension_dict
        
        # Convert to array!
        self.label_bin_df = self.label_bin_df.values

        #d = {"label_bin_dict": self.combined_df[0]}
        #df2 = pd.DataFrame(data = d)
        #df2.to_csv("results/test7.csv",",")
        #self.label_bin_df.to_csv("results/test7.csv",",")
        #np.savetxt("results/test.csv",self.label_bin_df,fmt="%d",delimiter=",")
        return self


    def create_pipeline_abs(self):
        """
        """
        self.clf_names = ["MNB", "BNB", "LRL1", "LRL2", "SVCL1", "SVCL2"]
        clfs = [
            MultinomialNB(alpha=0.01),
            BernoulliNB(alpha=0.01),
            LogisticRegression(C=100, penalty="l1", class_weight="auto"),
            LogisticRegression(C=10, penalty="l2", class_weight="auto"), 
            LinearSVC(C=10, penalty="l1", class_weight="auto", dual=False),
            LinearSVC(C=1, penalty="l2", class_weight="auto", dual=False)]

        ovr_clfs = [OneVsRestClassifier(clf) for clf in clfs]
        self.pipeline = [Pipeline([
                        ("vect", TfidfVectorizer(tokenizer=process_text,
                                                 min_df=3,
                                                 stop_words=self.stopwords,
                                                 sublinear_tf=True)),
                        ("ovr", clf)]) for clf in ovr_clfs]

        # So we can pick label dims!
        self.n_feature = [len(self.label_dimension_dict[col]) for col in self.column_name]
        self.dimension_end = dict(zip(self.column_name, np.cumsum(self.n_feature)))
        self.dimension_beg = {col:self.dimension_end[col] - len(self.label_dimension_dict[col]) for col in self.column_name}
        return self


    def run_grid_search_abs(self):
        """
        """
        for i in range(len(self.pipeline)):
            self.test = self.pipeline[i].fit(self.train_data["Abstract Text"], self.train_label)
            self.test2 = self.pipeline[i].predict(self.test_data["Abstract Text"])
            np.save("results/true_{0}.npy".format(i), self.test)

            if not os.path.exists("results/"):
                os.mkdir("results/")
            np.save("results/conf_{0}.npy".format(i), self.test2)
        return self


    """++++++++++++++++++++++++++++++++++++++++++++++"""  # Well this is informative...
    def create_2013_pipeline(self, alpha_param):
        """
        """
        self.clf_names = ["BNB"]
        clfs = [BernoulliNB(alpha=alpha_param)]
        # LinearSVC(penalty = "l2", class_weight="auto", dual=False, C=1.0)]

        ovr_clfs = [OneVsRestClassifier(clf) for clf in clfs]

        # "vect", TfidfVectorizer(min_df=3, stop_words=self.stopwords, sublinear_tf=True))
        self.pipeline = [Pipeline([("vect", CountVectorizer(binary=True)),
                                   ("ovr", clf)]) for clf in ovr_clfs]

        self.n_feature = [len(self.label_dimension_dict[col]) for col in self.column_name]
        self.dimension_end = dict(zip(self.column_name, np.cumsum(self.n_feature)))
        self.dimension_beg = {col:self.dimension_end[col] - len(self.label_dimension_dict[col]) for col in self.column_name}

        self.pipeline[0].fit(self.train_data["Abstract Text"], self.train_label)
        self.test_pred = self.pipeline[0].predict(self.test_data["Abstract Text"])
        # self.test_f1 = metrics.f1_score(self.test_label, self.test_pred, average="micro")
        return self.test_f1


    def create_pipeline(self):
        """
        All processing will be done with pipelines to make things easier.
        """
        print("Creating pipeline...")

        # Classifier names, arbitrary
        self.clf_names = ["MNB", "BNB", "LRL1", "LRL2", "SVCL1", "SVCL2"]
        
        # Classifiers used in the experiment
        clfs = [
            MultinomialNB(),
            BernoulliNB(),
            LogisticRegression(penalty="l1", class_weight="auto"),
            LogisticRegression(penalty="l2", class_weight="auto"), 
            LinearSVC(penalty="l1", class_weight="auto", dual=False),
            LinearSVC(penalty="l2", class_weight="auto", dual=False),
            ]

        # Generates one vs rest classifiers for each classifier
        ovr_clfs = [OneVsRestClassifier(clf) for clf in clfs]

        # Create pipeline consisting of the vectorizer and one vs rest classifiers
        self.pipeline = [Pipeline([
                            ("vect", TfidfVectorizer(min_df=3,
                                                     stop_words=self.stopwords,
                                                     sublinear_tf=True)),
                            ("ovr", clf)]) for clf in ovr_clfs]
        
        # Parameters to grid search over. Look at the individual classifiers
        # for details
        pipelines_parameters = [
            {"ovr__estimator__alpha": [0.01, 0.1, 1, 10]},
            {"ovr__estimator__alpha": [0.01, 0.1, 1, 10]}, 
            {"ovr__estimator__C": [0.1, 1, 10, 100]},
            {"ovr__estimator__C": [0.01, 0.1, 1, 10]},
            {"ovr__estimator__C": [0.01, 0.1, 1, 10]},
            {"ovr__estimator__C": [0.01, 0.1, 1, 10]}]

        # Pass above list of params to the pipeline
        self.pipelines_parameters = dict(zip(self.clf_names, pipelines_parameters))

        # Define grid searches and 10-Fold validation for the pipeline
        self.grid_searches = [
            {"grid_search": GridSearchCV(pl, param_grid=pp, 
                                         cv=KFold(len(self.train_data["Abstract Text"]),
                                                  n_folds=10, shuffle=True),
                                         scoring="f1_micro", n_jobs=-1, verbose=1)} 
            for pl, pp in zip(self.pipeline, pipelines_parameters)
            ]

        # Variable to hold our clf names and grid search stuff
        self.estimators = dict(zip(self.clf_names, self.grid_searches))

        # So we can pick label dims!
        self.n_feature = [len(self.label_dimension_dict[col]) for col in self.column_name]
        self.dimension_end = dict(zip(self.column_name, np.cumsum(self.n_feature)))
        self.dimension_beg = {col:self.dimension_end[col]-len(self.label_dimension_dict[col]) for col in self.column_name}


    def compute_2013_f1(self, label_dimension="all"):
        """
        """
        if label_dimension=="all":
            return metrics.f1_score(self.test_label, self.test_pred, average="micro")
        else:
            label_index_end = self.dimension_end[label_dimension]
            label_index_beg = self.dimension_beg[label_dimension]
            return metrics.f1_score(self.test_label[:, label_index_beg:label_index_end], 
                                    self.test_pred[:, label_index_beg:label_index_end],
                                    average="micro")


    def get_2013_f1s(self, run_num):
        """
        """
        ary = np.empty([1, len(self.column_name2)])
        y = 0
        for lbl in self.column_name2:
            val = self.compute_2013_f1(lbl)
            ary[0, y] = val
            y += 1
        np.save("results/f1_run{0}".format(run_num), ary)


    def compute_f1(self, clf_name, label_dimension="all"):
        """
        """
        label_pred = self.estimators[clf_name]["label_pred"]
        
        if label_dimension=="all":
            return metrics.f1_score(self.test_label, label_pred,
                                    average="micro")
        else:
            # Here we grab only the specific label dimension
            # This can be done because we generated those indices
            label_index_end = self.dimension_end[label_dimension]
            label_index_beg = self.dimension_beg[label_dimension]
            return metrics.f1_score(self.test_label[:, label_index_beg:label_index_end], 
                                    label_pred[:, label_index_beg:label_index_end],
                                    average="micro")
        

    def run_grid_search(self):
        """
        Grid search over params to get best ones.
        """
        for clf_name, clf in self.estimators.items():
            #print ("*** Grid Search " + clf_name + " ****")
            clf["grid_search"].fit(self.train_data["Abstract Text"], self.train_label)
            clf["label_pred"] = clf["grid_search"].predict(self.test_data["Abstract Text"])
        return self


    def get_f1s(self, run_num):
        """
        Generates F1 scores and saves them.
        """
        ary = np.empty([6, len(self.column_name2)])
        x = 0
        for clf in self.clf_names:
            y = 0
            for lbl in self.column_name2:
                val = self.compute_f1(clf, lbl)
                ary[x, y] = val
                y += 1
            x += 1
        np.save("results/f1_run{0}".format(run_num), ary)
        return self


    def nonstop_word_count(self, text, dic):
        """
        Counts the number of words (that aren't stop words) in a body of text.
        """
        words = text.split()
        for word in words:
            if word not in self.stopwords:
                if word in dic:
                    dic[word] = dic.get(word, 0) + 1
                else:
                    dic[word] = 1
        return dic


    def write_nonstop_word_count_per_article(self):
        """
        Counts the number of words (that aren't stop words) across all
        articles.
        """
        for f in self.filenames:
            dic = defaultdict(int)
            dic = self.nonstop_word_count(self.text_corpus[f], dic)
            df = pd.DataFrame(data=dic.items())
            if not os.path.exists("./wordCount/"):
                os.mkdir("./wordCount/")
            df.to_csv("./wordCount/{0}.csv".format(f), sep=",")
        

    def total_nonstop_word_count(self):
        """
        Counts the number of words (that aren't stop words) across all articles.
        """
        dic = defaultdict(int)
        for f in self.filenames:
            dic = self.nonstop_word_count(self.text_corpus[f], dic)
        return dic


    def word_list(self):
        """
        Returns a list with the number of words in each abstract.
        """
        word_list = []
        word_list = [len(abstract.split()) for abstract in self.label_df["Abstract Text"]]
        # for abs in self.label_df["Abstract Text"]:
        #     words = abs.split()
        #     word_list.append(len(words))
        return word_list


    def unique_word_list(self):
        """
        Returns a list with the number unique words in each abstract.
        """
        word_list = []
        word_list = [len(set(abstract.split())) for abstract in self.label_df["Abstract Text"]]
        
        # for abs in self.label_df["Abstract Text"]:
        #     words = abs.split()
        #     words_set = set(words)
        #     word_list.append(len(words_set))
        return word_list


    def do_confs(self, c_run):
        """
        """
        for clf in self.clf_names:
            for lbl_dim in self.column_name:
                self.conf(clf, lbl_dim, c_run)
        return self


    def do_confs_abs(self, c_run):
        """
        """
        for clf in self.clf_names:
            for lbl_dim in self.column_name:
                self.abs_conf(clf, lbl_dim, c_run)
        return self


    def abs_conf(self, clf_name, label_dimension, c_run):
        """
        """
        label_pred = self.test2
        label_index_end = self.dimension_end[label_dimension]
        label_index_beg = self.dimension_beg[label_dimension]
        subset_true = self.test_label[:, label_index_beg:label_index_end]
        
        # Make sure shape is the same
        if "Paradigm" in label_dimension:
            print(subset_true.shape)

        subset_pred = label_pred[:, label_index_beg:label_index_end]
        conf_array = np.empty(shape=subset_true.shape)
        for (x, y), value in np.ndenumerate(subset_true):
            # true negative
            if subset_true[x, y]==0 and subset_pred[x, y]==0:
                conf_array[x, y] = 1
            # false positive
            elif subset_true[x, y]==0 and subset_pred[x, y]==1:
                conf_array[x, y] = 2
            # false negative
            elif subset_true[x, y]==1 and subset_pred[x, y]==0:
                conf_array[x, y] = 3
            # true positive
            elif subset_true[x, y]==1 and subset_pred[x, y]==1:
                conf_array[x, y] = 4

        lbls = sorted(list(self.label_dimension_dict[label_dimension]))
        if not os.path.exists("results/heatmaps/"):
            os.mkdir("results/heatmaps/")

        #np.save("results/heatmaps/"+clf_name+"_"+label_dimension+"_"+str(c_run)+".csv",conf_array)
        print("Writing results/heatmaps/{0}_{1}_{2}.csv".format(clf_name, label_dimension, c_run))
        np.savetxt("results/heatmaps/{0}_{1}_{2}.csv".format(clf_name, label_dimension, c_run),
                   conf_array, fmt="%d", delimiter=",")
        np.savetxt("results/heatmaps/{0}_{1}_{2}_true.csv".format(clf_name, label_dimension, c_run),
                   subset_true, fmt="%d", delimiter=",")
        np.savetxt("results/heatmaps/{0}_{1}_{2}_pred.csv".format(clf_name, label_dimension, c_run),
                   subset_pred, fmt="%d", delimiter=",")
        with open("results/heatmaps/{0}_{1}_label_{2}.txt".format(clf_name, label_dimension, c_run), "w") as fo:
            for item in lbls:
                fo.write(item + "\n")
        return self


    def conf(self, clf_name, label_dimension, c_run):
        """
        """
        label_pred = self.estimators[clf_name]["label_pred"]
        label_index_end = self.dimension_end[label_dimension]
        label_index_beg = self.dimension_beg[label_dimension]
        subset_true = self.test_label[:, label_index_beg:label_index_end]
        subset_pred = label_pred[:, label_index_beg:label_index_end]
        conf_array = np.empty(shape=subset_true.shape)
        for (x, y), value in np.ndenumerate(subset_true):
            # true negative
            if subset_true[x, y]==0 and subset_pred[x, y]==0:
                conf_array[x, y] = 1
            # false positive
            elif subset_true[x, y]==0 and subset_pred[x, y]==1:
                conf_array[x, y] = 2
            # false negative
            elif subset_true[x, y]==1 and subset_pred[x, y]==0:
                conf_array[x, y] = 3
            # true positive
            elif subset_true[x, y]==1 and subset_pred[x, y]==1:
                conf_array[x, y] = 4
        lbls = sorted(list(self.label_dimension_dict[label_dimension]))

        if not os.path.exists("results/heatmaps/"):
            os.mkdir("results/heatmaps/")

        #np.save("results/heatmaps/"+clf_name+"_"+label_dimension+"_"+str(c_run)+".csv",conf_array)
        np.savetxt("results/heatmaps/{0}_{1}_{2}.csv".format(clf_name, label_dimension, c_run),
                   conf_array, fmt="%d", delimiter=",")
        np.savetxt("results/heatmaps/{0}_{1}_{2}_true.csv".format(clf_name, label_dimension, c_run),
                   subset_true, fmt="%d", delimiter=",")
        np.savetxt("results/heatmaps/{0}_{1}_{2}_pred.csv".format(clf_name, label_dimension, c_run),
                   subset_pred, fmt="%d", delimiter=",")
        with open("results/heatmaps/{0}_{1}_label_{2}.txt".format(clf_name, label_dimension, c_run), "w") as fo:
            for item in lbls:
                fo.write(item + "\n")


    def get_params(self, run_num):
        """
        """
        p_alpha = "ovr__estimator__alpha"
        p_c = "ovr__estimator__C"
        winning_params = []
        winning_vals = []

        # These clfs have alpha param
        for clf in ["MNB", "BNB"]:
            best_param = self.estimators[clf]["grid_search"].best_params_
            param_val = best_param.get(p_alpha)
            winning_params.append(p_alpha)
            winning_vals.append(param_val)

        # These clfs has c param
        for clf in ["LRL1", "LRL2", "SVCL1", "SVCL2"]:
            best_param = self.estimators[clf]["grid_search"].best_params_
            param_val = best_param.get(p_c)
            winning_params.append(p_c)
            winning_vals.append(param_val)

        with open("results/best_params_{0}.txt".format(run_num), "w") as fo:
            for item in range(0, 6):
                fo.write(str(winning_vals[item]))
                fo.write(" ")

        return self


    def get_coeff_vectors(self):
        """
        Coeff vectors
        """
        for clf_i in range(0, 6):
            coef_vect = self.pipeline[clf_i].steps[1][1].coef_
            #intercept = self.pipeline[clf_i].steps[1][1].intercept_
            feature_list = self.pipeline[clf_i].steps[0][1].get_feature_names()
            np.savetxt("results/coef_vect_{0}.csv".format(clf_i),
                       coef_vect, delimiter=",")
            with open("results/coef_names_{0}.txt".format(clf_i), "w") as fo:
                for item in feature_list:
                    fo.write(item + "\n")

            #np.savetxt("results/coef_list_"+str(clf_i)+".csv",feature_list,delimiter=",")
        return self


    def get_246(self, lbl):
        """
        """
        beg_index = athena.label_dimesion_beg[lbl]
        end_index = athena.label_dimesion_end[lbl]
        print beg_index
        print end_index


    def pickle_pipeline(self, run):
        """
        """
        with open("results/{0}_pipeline.p".format(run), "wb" ) as fo:
            pickle.dump(self.pipeline, fo)


def run_2013_abstracts(alpha_param):
    """
    Replicating Matt's results from 2013 (now with 100% more count
    vectorization).
    """
    # 2013 Test run
    for run in range(0, 10):
        # All same preprocessing/data stuff as normal run
        athena = Athena()
        athena.read_text()
        athena.read_stopwords()
        athena.read_meta_data()
        athena.combine_meta_data()
        athena.combine_data()
        athena.binarize()
        athena.split_data()
        
        # Now we use a separate pipeline with BNB, word count vect instead of tfidf
        athena.create_2013_pipeline(alpha_param)
        
        # Also separate way to get f1s obviously
        athena.get_2013_f1s(run)


# Program main functions
if __name__ == "__main__":
    #run_2013_abstracts(0.1)
    #This if for the 100% test 100% training run
    run = 0
    athena = Athena()
    athena.read_text()
    athena.read_stopwords()    
    athena.read_meta_data()
    athena.combine_meta_data()    
    athena.combine_data()
    athena.binarize()
    athena.split_data_abs()
    athena.create_pipeline_abs()
    athena.run_grid_search_abs()
    #athena.do_confs_abs(run)
    #athena.get_coeff_vectors()
    athena.pickle_pipeline("methods")

    """
    for i in range(0, 6):
        vocab = athena.pipeline[i].steps[0][1].vocabulary_
        pickle.dump(vocab,open("results/vocab_{0}.p".format(i), "wb"))
    dim = athena.label_dimension_dict
    dim_b = athena.dimension_beg
    dim_e = athena.dimension_end
    pickle.dump(dim, open("results/label_dimension_dict.p", "wb"))
    pickle.dump(dim_b, open("results/dim_beg.p", "wb"))
    pickle.dump(dim_e, open("results/dim_end.p", "wb"))
    """

    """
    # Normal Run
    for run in range(0, 10):
        print("Run " + str(run))
        athena = Athena()
        athena.read_text()
        athena.read_stopwords()
        athena.read_meta_data()
        athena.combine_meta_data()
        athena.combine_data()
        athena.binarize()
        athena.split_data()
        athena.create_pipeline()
        athena.run_grid_search()
        athena.get_f1s(run)
        athena.do_confs(run)
        athena.get_params(run)
    """
