import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from skrebate import ReliefF
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from .utils import plot_feature_ranking, plot_correlation_heatmap

class FeatureSelection:

    def correlation_matrix(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:

        X_encoded = pd.get_dummies(X)
        # Combine features + target
        df = pd.concat([X_encoded, y], axis=1)
        # Keep only numeric columns
        corr_matrix = df.corr()
        # Correlation with target
        plot_correlation_heatmap(corr_matrix, title="Correlation Heatmap between features")

        corr_target = (
            corr_matrix[y.name]
            .sort_values(ascending=False)
        )
        # Remove target itself
        ranking = corr_target.drop(y.name)
        plot_feature_ranking(ranking, title="Correlation Matrix", save_path="plots/corr_matrix_ranking.png")
        return ranking

    @staticmethod
    def chi_square_selection(X: pd.DataFrame, y: pd.Series, p_value_threshold: float = 0.05):
        # tengo solo colonne numeriche
        #X_numeric = X.select_dtypes(include=np.number)
        X_encoded = pd.get_dummies(X)
        # applico test Chi-quadrato
        chi_scores, p_values = chi2(X_encoded, y)

        results = pd.DataFrame({
            "Feature": X_encoded.columns,
            "Chi2 Score": chi_scores,
            "p-value": p_values,
        })

        print(results.sort_values("Chi2 Score", ascending=False))
        results = results[results["p-value"] < p_value_threshold]

        selected_features = results["Feature"]

        final_df = X_encoded[selected_features].copy()
        final_df[y.name] = y
        plot_feature_ranking(results, title="Chi-2 Analysis", save_path="plots/chi2_ranking.png")

        return final_df

    @staticmethod
    def information_gain_selection( X: pd.DataFrame, y: pd.Series, threshold: float = 0.02):
        # include TUTTE le feature (numeriche + categoriche)
        X_encoded = pd.get_dummies(X)
        mi = mutual_info_classif(X_encoded, y, random_state=42)

        scores = pd.Series(mi, index=X_encoded.columns)
        plot_feature_ranking( scores, title="Mutual Information")

        print(scores.sort_values(ascending=False))
        selected = scores[scores > threshold].index

        final_df = X_encoded[selected].copy()
        final_df[y.name] = y.values

        return final_df

    #l'idea è di applicarela PCA su solo alcune feature
    def PCA_selection(self, df: pd.DataFrame, target_col: str):
        y = df[target_col]
        X = df.drop(columns=[target_col])

        X_numeric = X.select_dtypes(include=np.number)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)

        pca = PCA()
        X_pca_transformed = pca.fit_transform(X_scaled)

        return pd.DataFrame(X_pca_transformed)

    def Relief_selection(self, X: pd.DataFrame, y: pd.Series):
        X_encoded = pd.get_dummies(X)

        X_encoded = X_encoded.astype(np.float64)

        relief = ReliefF(n_neighbors=100)
        relief.fit(X_encoded.values, y)

        scores = pd.Series(relief.feature_importances_, index=X.columns)
        ranking = scores.sort_values(ascending=False)

        print(ranking)
        plot_feature_ranking( ranking, title="ReliefF Ranking")
        top_features = ranking.head(5).index
        X_selected = X[top_features]
        print(X_selected.head())
        return X_selected

    @staticmethod
    def random_forest_selection(X: pd.DataFrame, y: pd.Series, threshold: float = 0.01):
        # encoding categoriche
        X_encoded = pd.get_dummies(X, drop_first=False)

        # modello Random Forest
        rf = RandomForestClassifier( n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_encoded, y)

        # importance scores
        scores = pd.Series( rf.feature_importances_, index=X_encoded.columns)

        ranking = scores.sort_values(ascending=False)

        print(ranking)
        plot_feature_ranking(ranking, title="Random Forest Importance")
        # selezione feature importanti
        selected_features = ranking[ranking > threshold].index
        # dataset finale
        final_df = X_encoded[selected_features].copy()
        final_df[y.name] = y.values
        return final_df

    @staticmethod
    def autoencoder_selection(
            X: pd.DataFrame,
            encoding_dim: int = 16,
            epochs: int = 6,
            batch_size: int = 32,
    ):
        # encoding categoriche
        X_encoded = pd.get_dummies(X)

        # scaling
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X_encoded)

        input_dim = X_scaled.shape[1]

        # ----- AUTOENCODER -----

        input_layer = Input(shape=(input_dim,))

        # encoder
        encoded = Dense(64, activation="relu")(input_layer)
        encoded = Dense(encoding_dim, activation="relu")(encoded)

        # decoder
        decoded = Dense(64, activation="relu")(encoded)
        decoded = Dense(input_dim, activation="sigmoid")(decoded)

        autoencoder = Model(input_layer, decoded)

        encoder = Model(input_layer, encoded)

        autoencoder.compile(
            optimizer=Adam(),
            loss="mse",
        )

        autoencoder.fit(
            X_scaled,
            X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=1,
        )

        # feature extraction
        X_latent = encoder.predict(X_scaled)

        latent_df = pd.DataFrame(
            X_latent,
            columns=[
                f"AE_Feature_{i}"
                for i in range(encoding_dim)
            ]
        )

        return latent_df

    @staticmethod
    def xgboost_selection(X: pd.DataFrame, y: pd.Series, threshold: float = 0.01):
        # encoding categoriche
        X_encoded = pd.get_dummies(X)

        # modello XGBoost
        model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        model.fit(X_encoded, y_encoded)

        # importance
        scores = pd.Series( model.feature_importances_, index=X_encoded.columns)

        ranking = scores.sort_values( ascending=False)
        plot_feature_ranking( ranking, title="XGBoost Importance")
        print(ranking)

        # selezione feature
        selected_features = ranking[ranking > threshold].index

        final_df = X_encoded[selected_features].copy()
        final_df[y.name] = y.values
        return final_df

    @staticmethod
    def catboost_selection( X: pd.DataFrame, y: pd.Series, threshold: float = 0.01):
        # individua colonne categoriche
        categorical_cols = X.select_dtypes( include=["object", "category"]).columns.tolist()

        # indici colonne categoriche
        cat_indices = [
            X.columns.get_loc(col)
            for col in categorical_cols
        ]

        # modello CatBoost
        model = CatBoostClassifier( iterations=100, random_state=42, verbose=0,)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        model.fit( X, y_encoded, cat_features=cat_indices)

        # importance
        scores = pd.Series( model.feature_importances_, index=X.columns)

        ranking = scores.sort_values(ascending=False)
        plot_feature_ranking( ranking, title="CatBoost Importance")
        print(ranking)

        # selezione feature
        selected_features = ranking[ ranking > threshold].index
        final_df = X[ selected_features].copy()
        final_df[y.name] = y.values
        return final_df