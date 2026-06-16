"""
preprocessing.py

Handles:
1. Data Loading
2. Data Inspection
3. Missing Values
4. Duplicate Removal
5. Label Encoding
6. Outlier Detection
7. Feature Scaling
8. Saving Clean Data
9. Saving Scaler
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class CustomerPreprocessor:

    def __init__(self, file_path):

        self.file_path = file_path
        self.df = None

        self.gender_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    # -----------------------------

    def load_data(self):

        self.df = pd.read_csv(self.file_path)

        print("\nDataset Loaded Successfully")
        print(self.df.head())

        return self.df

    # -----------------------------

    def basic_info(self):

        print("\nShape:")
        print(self.df.shape)

        print("\nColumns:")
        print(self.df.columns.tolist())

        print("\nData Types:")
        print(self.df.dtypes)

        print("\nMissing Values:")
        print(self.df.isnull().sum())

        print("\nDuplicate Rows:")
        print(self.df.duplicated().sum())

    # -----------------------------

    def remove_duplicates(self):

        before = len(self.df)

        self.df.drop_duplicates(inplace=True)

        after = len(self.df)

        print(f"\nDuplicates Removed : {before-after}")

    # -----------------------------

    def handle_missing_values(self):

        numeric_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        categorical_columns = self.df.select_dtypes(
            exclude=np.number
        ).columns

        for col in numeric_columns:

            self.df[col].fillna(
                self.df[col].median(),
                inplace=True
            )

        for col in categorical_columns:

            self.df[col].fillna(
                self.df[col].mode()[0],
                inplace=True
            )

        print("\nMissing Values Handled")

    # -----------------------------

    def encode_gender(self):

        if "Gender" in self.df.columns:

            self.df["Gender"] = self.gender_encoder.fit_transform(
                self.df["Gender"]
            )

            print("\nGender Encoded")

            print(
                dict(
                    zip(
                        self.gender_encoder.classes_,
                        self.gender_encoder.transform(
                            self.gender_encoder.classes_
                        ),
                    )
                )
            )

    # -----------------------------

    def detect_outliers(self):

        print("\nOutlier Summary")

        numeric_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        for col in numeric_columns:

            Q1 = self.df[col].quantile(0.25)

            Q3 = self.df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR

            upper = Q3 + 1.5 * IQR

            count = self.df[
                (self.df[col] < lower)
                | (self.df[col] > upper)
            ].shape[0]

            print(f"{col} : {count}")

    # -----------------------------

    def cap_outliers(self):

        numeric_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        for col in numeric_columns:

            Q1 = self.df[col].quantile(0.25)

            Q3 = self.df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR

            upper = Q3 + 1.5 * IQR

            self.df[col] = np.where(
                self.df[col] < lower,
                lower,
                self.df[col]
            )

            self.df[col] = np.where(
                self.df[col] > upper,
                upper,
                self.df[col]
            )

        print("\nOutliers Capped")

    # -----------------------------

    def scale_features(self):

        features = [
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)",
        ]

        scaled = self.scaler.fit_transform(
            self.df[features]
        )

        scaled_df = pd.DataFrame(
            scaled,
            columns=features
        )

        joblib.dump(
            self.scaler,
            "models/scaler.pkl"
        )

        print("\nScaler Saved")

        return scaled_df

    # -----------------------------

    def save_clean_data(self):

        self.df.to_csv(
            "data/cleaned_customers.csv",
            index=False
        )

        print("\nClean Dataset Saved")

    # -----------------------------

    def get_dataframe(self):

        return self.df