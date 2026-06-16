from src.preprocessing import CustomerPreprocessor
from src.eda import CustomerEDA
from src.clustering import CustomerClustering

# -------------------------

preprocessor = CustomerPreprocessor(
    "data/Mall_Customers.csv"
)

preprocessor.load_data()

preprocessor.basic_info()

preprocessor.handle_missing_values()

preprocessor.remove_duplicates()

preprocessor.encode_gender()

preprocessor.detect_outliers()

preprocessor.cap_outliers()

scaled_df = preprocessor.scale_features()

preprocessor.save_clean_data()

# -------------------------

df = preprocessor.get_dataframe()

eda = CustomerEDA(df)

eda.run_complete_eda()

# -------------------------

cluster = CustomerClustering(
    df,
    scaled_df
)

cluster.elbow_method()

best_k = cluster.silhouette_analysis()

cluster.train_model(5)

print("\nUsing Business Optimal K = 5")

cluster.pca_visualization()

cluster.interactive_plot()

cluster.cluster_summary()

cluster.save_clustered_data()