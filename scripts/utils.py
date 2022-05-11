import os
from tkinter import W
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class DataLoader:
    def __init__(self, dir_name, file_name):
        self.dir_name = dir_name
        self.file_name = file_name
        self.cwd = os.getcwd()

    def read_csv(self):
        os.chdir(self.dir_name)
        df = pd.read_csv(self.file_name)
        os.chdir(self.cwd)
        return df

    def read_excel(self):
        os.chdir(self.dir_name)
        df = pd.read_excel(self.file_name, engine='openpyxl')
        os.chdir(self.cwd)
        return df


class Plotters:
    def __init__(self, w, h) -> None:
        self.w = w
        self.h = h

    def plot_hist(self, df: pd.DataFrame, column: str, color: str) -> None:
        sns.displot(data=df, x=column, color=color,
                    kde=True, height=self.h, aspect=2)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        plt.show()

    def plot_count(self, df: pd.DataFrame, column: str) -> None:
        plt.figure(figsize=(self.w, self.h))
        sns.countplot(data=df, x=column)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        plt.show()

    def plot_bar(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str) -> None:
        plt.figure(figsize=(self.w, self.h))
        sns.barplot(data=df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.show()

    def plot_heatmap(self, df: pd.DataFrame, title: str, cbar=False) -> None:
        plt.figure(figsize=(self.w, self.h))
        sns.heatmap(df, annot=True, cmap='viridis', vmin=0,
                    vmax=1, fmt='.2f', linewidths=.7, cbar=cbar)
        plt.title(title, size=18, fontweight='bold')
        plt.show()

    def plot_box(self, df: pd.DataFrame, x_col: str, title: str) -> None:
        plt.figure(figsize=(self.w, self.h))
        sns.boxplot(data=df, x=x_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.show()

    def plot_box_multi(self, df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
        plt.figure(figsize=(self.w, self.h))
        sns.boxplot(data=df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
        plt.figure(figsize=(self.w, self.h))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, style=style)
        plt.title(title, size=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def plot_pair(self, df: pd.DataFrame, title: str, hue: str) -> None:
        plt.figure(figsize=(self.w, self.h))
        sns.pairplot(df,
                     hue=hue,
                     diag_kind='kde',
                     plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                     height=4)
        plt.title(title, size=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def plot_top_n_counts(self, df: pd.DataFrame, column: str, top_n: int, title: str, color: str = "green") -> pd.Series:
        top_n_count = df[column].value_counts().nlargest(top_n)
        axis = top_n_count.plot.bar(
            color=color, title=title, fontsize=20, figsize=(self.w, self.h))
        return top_n_count


class CleanDataFrame:

    def isolate_relavant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        relevant_columns = ['Bearer Id',
                            'Dur. ms',
                            'MSISDN/Number',
                            'Handset Manufacturer',
                            'Handset Type',
                            'Activity Duration DL (ms)',
                            'Activity Duration UL (ms)',
                            'Social Media DL (Bytes)',
                            'Social Media UL (Bytes)',
                            'Google DL (Bytes)',
                            'Google UL (Bytes)',
                            'Email DL (Bytes)',
                            'Email UL (Bytes)',
                            'Youtube DL (Bytes)',
                            'Youtube UL (Bytes)',
                            'Netflix DL (Bytes)',
                            'Netflix UL (Bytes)',
                            'Gaming DL (Bytes)',
                            'Gaming UL (Bytes)',
                            'Other DL (Bytes)',
                            'Other UL (Bytes)',
                            'Total UL (Bytes)',
                            'Total DL (Bytes)',
                            ]
        return df[relevant_columns]

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.rename(columns={
            "Dur. (ms).1": "Dur. ms",
            "Dur. (ms)": "Dur. s"},
            inplace=True)

        return df

    def fix_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes in the tellco dataframe an casts columns to proper data types.
        Start and End -> from string to datetime.
        Bearer Id, IMSI, MSISDN, IMEI -> From number to string
        """
        datetime_columns = ['Start',
                            'End', ]
        string_columns = [
            'IMSI',
            'MSISDN/Number',
            'IMEI',
            'Bearer Id'
        ]
        df_columns = df.columns
        for col in string_columns:
            if col in df_columns:
                df[col] = df[col].astype(str)
        for col in datetime_columns:
            if col in df_columns:
                df[col] = pd.to_datetime(df[col])

        return df

    def percent_missing(self, df):
        """
        Print out the percentage of missing entries in a dataframe
        """
        # Calculate total number of cells in dataframe
        totalCells = np.product(df.shape)

        # Count number of missing values per column
        missingCount = df.isnull().sum()

        # Calculate total number of missing values
        totalMissing = missingCount.sum()

        # Calculate percentage of missing values
        print("The dataset contains", round(
            ((totalMissing/totalCells) * 100), 2), "%", "missing values.")

    def handle_missing_value(self, df: pd.DataFrame, verbose=True) -> pd.DataFrame:
        if verbose:
            self.percent_missing(df)
            print(df.isnull().sum())
        numericals = df.select_dtypes(include='number').columns.tolist()
        objects = df.select_dtypes(exclude=['number']).columns.tolist()
        # if numericals:
        numeric_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='median')),
            # ('scale', MinMaxScaler()),
            # ('normalize', Normalizer()),
        ])
        cleaned_numerical = pd.DataFrame(
            numeric_pipeline.fit_transform(df[numericals]))
        cleaned_numerical.columns = numericals

        # if objects:
        object_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent'))
        ])
        cleaned_object = pd.DataFrame(
            object_pipeline.fit_transform(df[objects]))
        cleaned_object.columns = objects

        # if cleaned_object and cleaned_numerical:
        cleaned_df = pd.concat([cleaned_object, cleaned_numerical], axis=1)
        if verbose:
            print(cleaned_df.info())
            print(
                "="*10, "missing values imputed, collumns scalled, and normalized", "="*10)
        return cleaned_df
