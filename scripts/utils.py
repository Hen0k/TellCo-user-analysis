import os
from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer


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


class Analysis:
    def get_univariate_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        numerical_columns = CleanDataFrame.get_numerical_columns(df)
        numericals = df[numerical_columns]
        descriptions = numericals.describe().transpose()

        modes = {}
        for col in numericals.columns:
            modes[col] = numericals[col].mode()[0]
        descriptions['mode'] = modes.values()

        descriptions['CoV'] = descriptions['std'].values / \
            descriptions['mean'].values
        descriptions['skew'] = numericals.skew()
        descriptions['kurtosis'] = numericals.kurtosis().values
        Q1 = numericals.quantile(0.25)
        Q3 = numericals.quantile(0.75)
        IQR = Q3 - Q1
        descriptions['iqr'] = IQR
        descriptions['missing_counts'] = numericals.isna().sum()

        return descriptions

    @staticmethod
    def get_top_ten(df: pd.DataFrame, column: str) -> pd.DataFrame:
        df.sort_values(column, ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df.head(10)
    
    def get_missing_entries_count(self, df: pd.DataFrame) -> Union[pd.Series, list]:
        cols_missing_val_count = df.isnull().sum()
        cols_missing_val_count = cols_missing_val_count[cols_missing_val_count!=0]
        cols_missing_val = cols_missing_val_count.index.values
        cols_missing_val_count

        return cols_missing_val_count, cols_missing_val


class CleanDataFrame:

    @staticmethod
    def get_numerical_columns(df: pd.DataFrame) -> list:
        numerical_columns = df.select_dtypes(include='number').columns.tolist()
        return numerical_columns

    @staticmethod
    def get_categorical_columns(df: pd.DataFrame) -> list:
        categorical_columns = df.select_dtypes(
            include=['object']).columns.tolist()
        return categorical_columns

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

    def get_mct(self, series: pd.Series, measure: str):
        """
        get mean, median or mode depending on measure
        """
        measure = measure.lower()
        if measure == "mean":
            return series.mean()
        elif measure == "median":
            return series.median()
        elif measure == "mode":
            return series.mode()[0]

    def replace_missing(self, df: pd.DataFrame, columns: str, method: str) -> pd.DataFrame:
        
        for column in columns:
            nulls = df[column].isnull()
            indecies = [i for i, v in zip(nulls.index, nulls.values) if v]
            replace_with = self.get_mct(df[column], method)
            df.loc[indecies, column] = replace_with
            

        return df
    
    
    def remove_null_row(self, df: pd.DataFrame, columns: str) -> pd.DataFrame:
        for column in columns:
            df = df[~ df[column].isna()]
        
        return df

    def handle_missing_value(self, df: pd.DataFrame, verbose=True) -> pd.DataFrame:
        if verbose:
            self.percent_missing(df)
            print(df.isnull().sum())
        numericals = self.get_numerical_columns(df)
        objects = self.get_categorical_columns(df)
        for col in objects:
            df = df[df[col] != "nan"]
            # df[col].fillna(df[col].mode(), inplace=True)
            # df[col].dropna(inplace=True)
        for col in numericals:
            df[col].fillna(df[col].median(), inplace=True)
        # if numericals:
        # numeric_pipeline = Pipeline(steps=[
        #     ('impute', SimpleImputer(strategy='median')),
        #     # ('scale', MinMaxScaler()),
        #     # ('normalize', Normalizer()),
        # ])
        # cleaned_numerical = pd.DataFrame(
        #     numeric_pipeline.fit_transform(df[numericals]))
        # cleaned_numerical.columns = numericals

        # # if objects:
        # object_pipeline = Pipeline(steps=[
        #     ('impute', SimpleImputer(strategy='most_frequent'))
        # ])
        # cleaned_object = pd.DataFrame(
        #     object_pipeline.fit_transform(df[objects]))
        # cleaned_object.columns = objects

        # # if cleaned_object and cleaned_numerical:
        # cleaned_df = pd.concat([cleaned_object, cleaned_numerical], axis=1)
        if verbose:
            print(df.info())
            print(
                "="*10, "missing values imputed, collumns scalled, and normalized", "="*10)

        return df

    def normal_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        scaller = StandardScaler()
        scalled = pd.DataFrame(scaller.fit_transform(
            df[self.get_numerical_columns(df)]))
        scalled.columns = self.get_numerical_columns(df)

        return scalled

    def minmax_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        scaller = MinMaxScaler()
        scalled = pd.DataFrame(
            scaller.fit_transform(
                df[self.get_numerical_columns(df)]),
            columns=self.get_numerical_columns(df)
        )

        return scalled

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        normalizer = Normalizer()
        normalized = pd.DataFrame(
            normalizer.fit_transform(
                df[self.get_numerical_columns(df)]),
            columns=self.get_numerical_columns(df)
        )

        return normalized
