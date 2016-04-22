import data_preparation
import feature_selection
import feature_extraction

if __name__ == '__main__':
    feature_extraction.extract_features()
    feature_selection.run_feature_selection()
