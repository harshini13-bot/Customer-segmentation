import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class CustomerClustering:

    def __init__(self, dataframe, scaled_data):

        self.df = dataframe
        self.scaled_data = scaled_data

        os.makedirs("models", exist_ok=True)
        os.makedirs("assets", exist_ok=True)

        self.model = None

    # -----------------------------------

    def elbow_method(self):

        wcss = []

        for k in range(1, 11):

            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            model.fit(self.scaled_data)

            wcss.append(model.inertia_)

        plt.figure(figsize=(8, 5))

        plt.plot(
            range(1, 11),
            wcss,
            marker="o"
        )

        plt.xlabel("Number of Clusters")

        plt.ylabel("WCSS")

        plt.title("Elbow Method")

        plt.savefig("assets/elbow_method.png")

        plt.show()

    # -----------------------------------

    def silhouette_analysis(self):

        print("\nSilhouette Scores\n")

        best_score = -1
        best_k = 2

        for k in range(2, 11):

          model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
          )

          labels = model.fit_predict(self.scaled_data)

          score = silhouette_score(
            self.scaled_data,
            labels
          )

          print(f"K={k} --> {score:.4f}")

          if score > best_score:

            best_score = score
            best_k = k

        print("\nBest K =", best_k)
        print("Best Score =", round(best_score,4))

        return best_k

    # -----------------------------------
    def train_model(self, n_clusters):

        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        labels = self.model.fit_predict(
            self.scaled_data
        )

        self.df["Cluster"] = labels

        # Segment Names
        cluster_names = {
            0: "Premium Customers",
            1: "Budget Customers",
            2: "Occasional Buyers",
            3: "Young High Spenders",
            4: "Loyal Customers"
        }

        self.df["Segment"] = self.df["Cluster"].map(cluster_names)
 
        joblib.dump(
            self.model,
            "models/kmeans_model.pkl"
        )

        print("\nModel Saved Successfully")

    # -----------------------------------

    def pca_visualization(self):

        pca = PCA(n_components=2)

        components = pca.fit_transform(
            self.scaled_data
        )

        pca_df = pd.DataFrame()

        pca_df["PC1"] = components[:, 0]

        pca_df["PC2"] = components[:, 1]

        pca_df["Cluster"] = self.df["Cluster"]

        plt.figure(figsize=(8, 6))

        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue="Cluster",
            palette="Set2",
            s=100
        )

        plt.title("PCA Cluster Visualization")

        plt.savefig("assets/pca_clusters.png")

        plt.show()

    # -----------------------------------

    def interactive_plot(self):

        pca = PCA(n_components=2)

        components = pca.fit_transform(
            self.scaled_data
        )

        plot_df = pd.DataFrame()

        plot_df["PC1"] = components[:, 0]

        plot_df["PC2"] = components[:, 1]

        plot_df["Cluster"] = self.df["Cluster"]

        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color=plot_df["Cluster"].astype(str),
            title="Interactive Customer Segmentation"
        )

        fig.write_html(
            "assets/interactive_clusters.html"
        )

        fig.show()

    # -----------------------------------

    def save_clustered_data(self):

        self.df.to_csv(
            "data/clustered_customers.csv",
            index=False
        )

        print("\nClustered Dataset Saved")

    # -----------------------------------

    def cluster_summary(self):

        print("\nCluster Summary\n")

        summary = (
            self.df.groupby("Cluster")
            .agg({
                "Age": "mean",
                "Annual Income (k$)": "mean",
                "Spending Score (1-100)": "mean",
                "CustomerID": "count"
            })
            .rename(columns={"CustomerID": "Customer Count"})
        )

        print(summary)