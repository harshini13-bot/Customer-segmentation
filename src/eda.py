import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


class CustomerEDA:

    def __init__(self, dataframe):

        self.df = dataframe

        os.makedirs("assets", exist_ok=True)

        sns.set_style("whitegrid")

    # --------------------------

    def dataset_info(self):

        print("\nShape")
        print(self.df.shape)

        print("\nColumns")
        print(self.df.columns.tolist())

        print("\nData Types")
        print(self.df.dtypes)

        print("\nStatistics")
        print(self.df.describe())

    # --------------------------

    def correlation_heatmap(self):

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            self.df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm"
        )

        plt.title("Correlation Heatmap")

        plt.savefig("assets/correlation_heatmap.png")

        plt.show()

    # --------------------------

    def age_distribution(self):

        plt.figure(figsize=(8, 5))

        sns.histplot(
            self.df["Age"],
            bins=20,
            kde=True
        )

        plt.title("Age Distribution")

        plt.savefig("assets/age_distribution.png")

        plt.show()

    # --------------------------

    def income_distribution(self):

        plt.figure(figsize=(8, 5))

        sns.histplot(
            self.df["Annual Income (k$)"],
            bins=20,
            kde=True
        )

        plt.title("Income Distribution")

        plt.savefig("assets/income_distribution.png")

        plt.show()

    # --------------------------

    def spending_distribution(self):

        plt.figure(figsize=(8, 5))

        sns.histplot(
            self.df["Spending Score (1-100)"],
            bins=20,
            kde=True
        )

        plt.title("Spending Score Distribution")

        plt.savefig("assets/spending_distribution.png")

        plt.show()

    # --------------------------

    def gender_count(self):

        plt.figure(figsize=(6, 5))

        sns.countplot(
            x="Gender",
            data=self.df
        )

        plt.title("Gender Count")

        plt.savefig("assets/gender_count.png")

        plt.show()

    # --------------------------

    def age_boxplot(self):

        plt.figure(figsize=(6, 5))

        sns.boxplot(
            y=self.df["Age"]
        )

        plt.title("Age Boxplot")

        plt.savefig("assets/age_boxplot.png")

        plt.show()

    # --------------------------

    def income_boxplot(self):

        plt.figure(figsize=(6, 5))

        sns.boxplot(
            y=self.df["Annual Income (k$)"]
        )

        plt.title("Income Boxplot")

        plt.savefig("assets/income_boxplot.png")

        plt.show()

    # --------------------------

    def spending_boxplot(self):

        plt.figure(figsize=(6, 5))

        sns.boxplot(
            y=self.df["Spending Score (1-100)"]
        )

        plt.title("Spending Score Boxplot")

        plt.savefig("assets/spending_boxplot.png")

        plt.show()

    # --------------------------

    def pair_plot(self):

        pair = sns.pairplot(
            self.df,
            hue="Gender",
            diag_kind="hist"
        )

        pair.fig.suptitle(
            "Pair Plot of Customer Data",
             y=1.02
        )

        pair.savefig("assets/pairplot.png")

    plt.show()

    # --------------------------

    def age_vs_income(self):

        plt.figure(figsize=(8, 5))

        sns.scatterplot(
            data=self.df,
            x="Age",
            y="Annual Income (k$)"
        )

        plt.title("Age vs Income")

        plt.savefig("assets/age_vs_income.png")

        plt.show()

    # --------------------------

    def income_vs_spending(self):

        plt.figure(figsize=(8, 5))

        sns.scatterplot(
            data=self.df,
            x="Annual Income (k$)",
            y="Spending Score (1-100)"
        )

        plt.title("Income vs Spending")

        plt.savefig("assets/income_vs_spending.png")

        plt.show()

    # --------------------------

    def age_vs_spending(self):

        plt.figure(figsize=(8, 5))

        sns.scatterplot(
            data=self.df,
            x="Age",
            y="Spending Score (1-100)"
        )

        plt.title("Age vs Spending")

        plt.savefig("assets/age_vs_spending.png")

        plt.show()

    # --------------------------

    def violin_plot(self):

        plt.figure(figsize=(8, 5))

        sns.violinplot(
            y=self.df["Spending Score (1-100)"]
        )

        plt.title("Violin Plot")

        plt.savefig("assets/violin_plot.png")

        plt.show()

    # --------------------------

    def interactive_income_spending(self):

        fig = px.scatter(
            self.df,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            color="Gender",
            hover_data=["Age"]
        )

        fig.write_html(
            "assets/interactive_income_spending.html"
        )

        fig.show()

    # --------------------------

    def interactive_age_income(self):

        fig = px.scatter(
            self.df,
            x="Age",
            y="Annual Income (k$)",
            color="Gender"
        )

        fig.write_html(
            "assets/interactive_age_income.html"
        )

        fig.show()

    # --------------------------

    def run_complete_eda(self):

        self.dataset_info()

        self.correlation_heatmap()

        self.age_distribution()

        self.income_distribution()

        self.spending_distribution()

        self.gender_count()

        self.age_boxplot()

        self.income_boxplot()

        self.spending_boxplot()

        self.pair_plot()

        self.age_vs_income()

        self.income_vs_spending()

        self.age_vs_spending()

        self.violin_plot()

        self.interactive_income_spending()

        self.interactive_age_income()

        print("\nEDA Completed Successfully")