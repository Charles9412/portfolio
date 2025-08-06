import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

def quick_eda(df, top_n=5, show_plots=True):
    """
    Prints a quick exploratory data summary of the dataframe and displays basic plots.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        top_n (int): Number of top values to show for categorical columns
        show_plots (bool): Whether to display histograms and pairplot
    """
    print(" --- Resultados Análisis Exploratorio ---\n")

    # Basic info
    print(" Shape:", df.shape)
    print("\nTipos de Columnas:")
    print(df.dtypes)
    
    print("\n Valores Faltantes:")
    print(df.isnull().sum())

    # Numeric summary
    print("\n Resumen:")
    display(df.describe().T)

    # Categorical summary
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

    if len(cat_cols) > 0:
        print("\n Columnas Categóricas/Booleanas:")
        for col in cat_cols:
            print(f"\n▶ {col}:")
            print(df[col].value_counts(dropna=False).head(top_n))
    else:
        print("\nℹ No se encontraron variables categóricas o Booleanas")

    # Plots
    if show_plots:
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns

        print("\n Histograms:")
        df[num_cols].hist(bins=20, figsize=(15, 10), edgecolor='black')
        plt.tight_layout()
        plt.show()

        if len(num_cols) > 1:
            print("\n🔗 Pairplot (subset of numerical features):")
            sns.pairplot(df[num_cols].sample(min(200, len(df))), diag_kind="kde")
            plt.show()

    print("\n Análisis Exploratorio Completado .\n")


def generate_mls_dataframe(n=100, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    df = pd.DataFrame({
        "numero_de_contrato": np.random.randint(10000, 99999, size=n),  # 5-digit random numbers
        # Transaccionales
        "colocacion_mxn": np.round(np.random.uniform(500, 10000, size=n), 2),
        "cargo_pct": np.round(np.random.uniform(5, 25, size=n), 2),
        "frecuencia_empeno_dias": np.random.randint(5, 180, size=n),
        "tipo_articulo": np.random.choice(["reloj", "alhaja", "celular", "electronico"], size=n),

        # Comportamiento Digital
        "onboarding": np.random.choice([1, 0], size=n),
        "pago_app_mxn": np.round(np.random.uniform(0, 5000, size=n), 2),
        "referenciacion": np.random.poisson(lam=2, size=n),
        "credito_digital": np.random.choice([1, 0], size=n)
    })

    return df


def compute_weighted_score(df, columns, weights, score_name="score"):
    """
    Compute a weighted score based on normalized values of selected columns.
    Categorical columns are temporarily label-encoded internally.
    """
    df = df.copy()
    normalized = pd.DataFrame()

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        if col not in weights:
            raise ValueError(f"No weight provided for column '{col}'.")

        col_data = df[col]

        # Encode non-numeric columns temporarily
        if not np.issubdtype(col_data.dtype, np.number):
            le = LabelEncoder()
            col_data = le.fit_transform(col_data)

        # Min-max normalization
        min_val = col_data.min()
        max_val = col_data.max()
        if min_val == max_val:
            normalized[col] = 0  # Avoid division by zero
        else:
            normalized[col] = (col_data - min_val) / (max_val - min_val)

    # Weighted sum
    df[score_name] = sum(normalized[col] * weights[col] for col in columns)
    return df

def segment_and_profile(df, features, n_clusters=3, cluster_col="cluster"):
    """
    Segments the dataframe using K-Means and returns the clustered DataFrame and the prototypes.

    Parameters:
        df (pd.DataFrame): DataFrame with scoring columns
        features (list): Columns to use for clustering (e.g., ["score_transaccional", "score_digital"])
        n_clusters (int): Number of clusters
        cluster_col (str): Name of the column to store the cluster labels

    Returns:
        tuple: (df_with_clusters, prototypes_df)
    """
    df = df.copy()
    X = df[features]

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    df[cluster_col] = kmeans.fit_predict(X)

    # Exclude identifier columns from profiling
    exclude_cols = [cluster_col, "numero_de_contrato"]
    profile_cols = [col for col in df.columns if col not in exclude_cols]

    # Profiling (only meaningful numeric columns)
    prototypes = df.groupby(cluster_col)[profile_cols].mean(numeric_only=True)

    return df, prototypes


def plot_clusters(df, x_col="score_transaccional", y_col="score_digital", cluster_col="cluster", title="Segmentación de Clientes", save_path=None):
    """
    Plots a 2D scatter plot of clusters with enhanced styling, including centroids.

    Parameters:
        df (pd.DataFrame): DataFrame with cluster assignments
        x_col (str): Name of the column for the X-axis
        y_col (str): Name of the column for the Y-axis
        cluster_col (str): Name of the cluster label column
        title (str): Title of the plot
        save_path (str): If provided, saves the plot to this path
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    n_clusters = df[cluster_col].nunique()
    palette = sns.color_palette("Set2", n_clusters)

    # Scatter plot of points
    scatter = sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=cluster_col,
        palette=palette,
        s=70,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85
    )

    # Calculate centroids
    centroids = df.groupby(cluster_col)[[x_col, y_col]].mean()

    # Plot centroids
    plt.scatter(
        centroids[x_col],
        centroids[y_col],
        s=200,
        c="red",
        marker="X",
        label="Centroides"
    )

    # Annotate centroids
    for i, (x, y) in centroids.iterrows():
        plt.text(x, y + 0.02, f"C{i}", fontsize=10, ha='center', va='bottom', weight='bold', color='black')

    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel(x_col.replace("_", " ").capitalize(), fontsize=12)
    plt.ylabel(y_col.replace("_", " ").capitalize(), fontsize=12)
    plt.legend(title="Cluster", loc="upper left", bbox_to_anchor=(1, 1))
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def recommend_by_cluster(cluster_id):
    """
    Returns a personalized recommendation based on the user's cluster.
    """
    recommendations = {
        0: "🎯 ¡Aprovecha nuestra promoción exclusiva en créditos digitales!",
        1: "📈 Tu historial de inversión es excelente. ¿Te interesa un rendimiento más alto?",
        2: "💡 ¿Sabías que puedes pagar tus empeños directamente desde la app?",
        3: "🎁 Invita a tus amigos y gana beneficios exclusivos por referenciación."
    }

    return recommendations.get(cluster_id, "🤖 Bienvenido. Explora nuestras opciones personalizadas para ti.")

def mock_recommendation(df, cluster_col="cluster", contrato_col="numero_de_contrato"):
    """
    Simulates a client using the app and returns their cluster info and recommendation.

    Parameters:
        df (pd.DataFrame): The full dataframe with cluster assignments and contract numbers
        cluster_col (str): Name of the column with cluster labels
        contrato_col (str): Name of the column with contract IDs

    Returns:
        tuple: (summary_text, recommendation_text)
    """
    client = df.sample(1).iloc[0]
    
    contrato = int(client[contrato_col])
    cluster = int(client[cluster_col])

    mensaje = recommend_by_cluster(cluster)

    resumen = f"""
    🆔 Contrato: {contrato}
    🧩 Cluster asignado: {cluster}
    📊 Score Transaccional: {client.get('score_transaccional', 0):.2f}
    📱 Score Digital: {client.get('score_digital', 0):.2f}
    """

    return resumen.strip(), mensaje

