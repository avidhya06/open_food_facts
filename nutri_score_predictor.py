import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import re


class NutriScorePredictor:
    def __init__(self, config):
        """
        Initializes the NutriScorePredictor with the given configuration.

        Args:
            config (dict): Configuration dictionary containing parameters for the predictor.
        """
        self.config = config
        self.model = None
        self.label_encoder = None
        self.num_imputer = None
        self.tfidf_vectorizer = None

    def load_data(self):
        """
        Loads data from a JSON Lines file into a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data, or None if an error occurs.
        """
        data = []
        file_path = self.config['JSONL_FILE_PATH']
        print(f"Loading data from {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON on line {i+1}")
            df = pd.DataFrame(data)
            print(f"Loaded {len(df)} records.")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        except Exception as e:
            print(f"An error occurred during file loading: {e}")
            return None

    def preprocess_data(self, df):
        """
        Prepares the data for model training by filtering, cleaning, and transforming it.

        Args:
            df (pd.DataFrame): Input DataFrame to preprocess.

        Returns:
            tuple: Processed numerical features (X_num), text features (X_text), and target variable (y).
        """
        print("Preprocessing data...")
        num_features = self.config['NUMERICAL_FEATURES']
        text_feature = self.config['TEXT_FEATURE']
        target = self.config['TARGET_VARIABLE']
        valid_labels = self.config['VALID_TARGET_LABELS']
        use_text = self.config['USE_INGREDIENTS_TEXT']

        # Select relevant columns
        cols_to_keep = num_features + [target]
        if use_text:
            cols_to_keep.append(text_feature)
        df_processed = df[cols_to_keep].copy()

        # Handle Target Variable
        df_processed.dropna(subset=[target], inplace=True)
        df_processed = df_processed[df_processed[target].isin(valid_labels)]
        print(f"Records after filtering for valid target '{target}': {len(df_processed)}")
        if len(df_processed) == 0:
            print("Error: No valid target data remaining after filtering.")
            return None, None, None

        # Handle Numerical Features
        for col in num_features:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Handle Text Feature (if used)
        if use_text:
            df_processed[text_feature] = df_processed[text_feature].fillna('')
            df_processed[text_feature] = df_processed[text_feature].apply(
                lambda x: re.sub(r'[^\w\s]', '', x.lower()) if isinstance(x, str) else ''
            )

        y = df_processed[target]
        X_num = df_processed[num_features]
        X_text = df_processed[text_feature] if use_text else None

        print(f"Preprocessing complete. Shape before split: {df_processed.shape}")
        return X_num, X_text, y

    def split_data(self, X_num, X_text, y):
        """
        Splits the data into training and testing sets.

        Args:
            X_num (pd.DataFrame): Numerical features.
            X_text (pd.Series or None): Text features, if applicable.
            y (pd.Series): Target variable.

        Returns:
            tuple: Training and testing sets for numerical features, text features, and target variable.
        """
        print(f"Splitting data (Test size: {self.config['TEST_SET_SIZE']})...")
        indices = X_num.index
        train_indices, test_indices = train_test_split(
            indices, test_size=self.config['TEST_SET_SIZE'], random_state=self.config['RANDOM_STATE'], stratify=y.loc[indices]
        )

        X_num_train, X_num_test = X_num.loc[train_indices], X_num.loc[test_indices]
        y_train, y_test = y.loc[train_indices], y.loc[test_indices]

        if self.config['USE_INGREDIENTS_TEXT'] and X_text is not None:
            X_text_train, X_text_test = X_text.loc[train_indices], X_text.loc[test_indices]
        else:
            X_text_train, X_text_test = None, None

        print(f"Train set size: {len(y_train)}, Test set size: {len(y_test)}")
        return X_num_train, X_num_test, X_text_train, X_text_test, y_train, y_test

    def transform_features(self, X_num_train, X_num_test, X_text_train, X_text_test):
        """
        Applies feature engineering and transformation to the data.

        Args:
            X_num_train (pd.DataFrame): Training numerical features.
            X_num_test (pd.DataFrame): Testing numerical features.
            X_text_train (pd.Series or None): Training text features, if applicable.
            X_text_test (pd.Series or None): Testing text features, if applicable.

        Returns:
            tuple: Transformed training and testing feature matrices.
        """
        print("Applying numerical imputation (median)...")
        self.num_imputer = SimpleImputer(strategy='median')
        X_num_train_imputed = self.num_imputer.fit_transform(X_num_train)
        X_num_test_imputed = self.num_imputer.transform(X_num_test)

        X_train_final = X_num_train_imputed
        X_test_final = X_num_test_imputed

        if self.config['USE_INGREDIENTS_TEXT'] and X_text_train is not None:
            print("Applying text vectorization (TF-IDF)...")
            self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            X_text_train_tfidf = self.tfidf_vectorizer.fit_transform(X_text_train)
            X_text_test_tfidf = self.tfidf_vectorizer.transform(X_text_test)

            print("Combining numerical and text features...")
            X_train_final = hstack([X_num_train_imputed, X_text_train_tfidf]).tocsr()
            X_test_final = hstack([X_num_test_imputed, X_text_test_tfidf]).tocsr()

        print(f"Final features shape (train): {X_train_final.shape}")
        print(f"Final features shape (test): {X_test_final.shape}")
        return X_train_final, X_test_final

    def encode_target(self, y_train, y_test):
        """
        Encodes the target labels into numerical format.

        Args:
            y_train (pd.Series): Training target labels.
            y_test (pd.Series): Testing target labels.

        Returns:
            tuple: Encoded training and testing target labels.
        """
        print("Encoding target labels...")
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        print(f"Target classes: {self.label_encoder.classes_}")
        return y_train_encoded, y_test_encoded

    def train_model(self, X_train, y_train):
        """
        Trains the RandomForestClassifier model on the training data.

        Args:
            X_train (array-like): Training feature matrix.
            y_train (array-like): Training target labels.
        """
        print("Training RandomForestClassifier model...")
        self.model = RandomForestClassifier(
            n_estimators=150,
            random_state=self.config['RANDOM_STATE'],
            n_jobs=-1,
            class_weight='balanced',
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=3
        )
        self.model.fit(X_train, y_train)
        print("Model training complete.")

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the trained model on the test set.

        Args:
            X_test (array-like): Testing feature matrix.
            y_test (array-like): Testing target labels.

        Prints:
            Accuracy and classification report of the model.
        """
        print("Evaluating model on the test set...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        print("\n--- Evaluation Results ---")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        print("------------------------")
