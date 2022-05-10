import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    def plot_hist(self, df: pd.DataFrame, column: str, color: str) -> None:
        plt.figure(figsize=(5, 3))
        fig, ax = plt.subplots(1, figsize=(12, 7))
        sns.histplot(data=df, x=column, color=color,
                     kde=True, height=7, aspect=2, bins=10)
        # sns.histplot(data=df, x=column, color=color, bins=15, stat='count')
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        plt.show()

    def plot_box(self, df: pd.DataFrame, x_col: str, title: str) -> None:
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x=x_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.show()

    def plot_box_multi(self, df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def plot_scatter(self, df:  pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
        plt.figure(figsize=(12, 7))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, style=style)
        plt.title(title, size=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def plot_heatmap(self, df: pd.DataFrame, title: str, cbar=False) -> None:
        plt.figure(figsize=(12, 7))
        sns.heatmap(df, annot=True, cmap='viridis', vmin=0,
                    vmax=1, fmt='.2f', linewidths=.7, cbar=cbar)
        plt.title(title, size=18, fontweight='bold')
        plt.show()
