import pandas as pd
import numpy as np
import os


def load_and_label_data(base_path):
    df_list = []

    # Load benign
    benign_file = os.path.join(base_path, "benign_traffic.csv")
    benign_df = pd.read_csv(benign_file)
    benign_df["attack_type"] = "benign"
    df_list.append(benign_df)

    # Load Gafgyt
    gafgyt_path = os.path.join(base_path, "gafgyt_attacks")
    for file in os.listdir(gafgyt_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(gafgyt_path, file))
            df["attack_type"] = "gafgyt_" + file.replace(".csv", "")
            df_list.append(df)

    # Load Mirai
    mirai_path = os.path.join(base_path, "mirai_attacks")
    for file in os.listdir(mirai_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(mirai_path, file))
            df["attack_type"] = "mirai_" + file.replace(".csv", "")
            df_list.append(df)

    # Combine all
    combined_df = pd.concat(df_list, ignore_index=True)
    print("Combined shape:", combined_df.shape)
    print(combined_df["attack_type"].value_counts())

    # Save combined data to a CSV file
    combined_df.to_csv("combined_data.csv", index=False)

    return combined_df


def feature_selection_extraction(df, k_best=50, pca_components=10):
    X = df.drop(columns=["attack_type"])
    y = df["attack_type"]

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Filter-based feature selection (ANOVA F-test)
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(score_func=f_classif, k=k_best)
    X_selected = selector.fit_transform(X_scaled, y_enc)

    # PCA for feature extraction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_selected, X_pca, y_enc, le, selector, pca


# === MAIN FUNCTION ===
if __name__ == "__main__":
    base_path = "."  # Adjust this path if needed
    df = load_and_label_data(base_path)
    X_selected, X_pca, y_enc, le, selector, pca = feature_selection_extraction(df)
