import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .utils import plot_feature_ranking

class FeatureSelection:
    """
    Classe per l'analisi e la selezione delle feature.
    I metodi sono stati ottimizzati per lavorare su dati già preprocessati (numerici).
    """

    def correlation_ranking(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calcola la correlazione di ogni feature con il target."""
        # Uniamo temporaneamente per calcolare la correlazione
        df = pd.concat([X, y], axis=1)
        corr_matrix = df.corr()
        
        # Correlazione con il target (valore assoluto per importanza)
        ranking = corr_matrix[y.name].abs().sort_values(ascending=False)
        ranking = ranking.drop(y.name) # Rimuoviamo il target stesso
        
        plot_feature_ranking(ranking, title="Correlation Ranking", save_path="plots/corr_ranking.png")
        return ranking

    def get_high_correlation_pairs(self, X: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """Identifica coppie di feature altamente correlate tra loro."""
        corr_matrix = X.corr()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        high_corr_pairs = upper_triangle.stack().reset_index()
        high_corr_pairs.columns = ["feature_1", "feature_2", "correlation"]
        
        high_corr_pairs = high_corr_pairs[high_corr_pairs["correlation"].abs() >= threshold]
        return high_corr_pairs.sort_values(by="correlation", ascending=False)

    @staticmethod
    def chi_square_scores(X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calcola i punteggi Chi-quadrato (solo per feature non negative)."""
        # Assicuriamoci che non ci siano valori negativi (comune dopo lo scaling)
        X_min = X.min().min()
        X_adj = X - X_min if X_min < 0 else X
        
        chi_scores, p_values = chi2(X_adj, y)
        scores = pd.Series(chi_scores, index=X.columns).sort_values(ascending=False)
        
        plot_feature_ranking(scores, title="Chi-2 Scores", save_path="plots/chi2_ranking.png")
        return scores

    @staticmethod
    def information_gain_scores(X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calcola la Mutual Information tra feature e target."""
        mi = mutual_info_classif(X, y, random_state=42)
        scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        
        plot_feature_ranking(scores, title="Mutual Information Scores", save_path="plots/mi_ranking.png")
        return scores

    @staticmethod
    def random_forest_importances(X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calcola l'importanza delle feature tramite Random Forest."""
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        scores = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        plot_feature_ranking(scores, title="Random Forest Importance", save_path="plots/rf_importance.png")
        return scores

    @staticmethod
    def xgboost_importances(X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calcola l'importanza delle feature tramite XGBoost."""
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier
        
        model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        model.fit(X, y_encoded)
        
        scores = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        plot_feature_ranking(scores, title="XGBoost Importance", save_path="plots/xgb_importance.png")
        return scores

    @staticmethod
    def catboost_importances(X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calcola l'importanza delle feature tramite CatBoost."""
        from sklearn.preprocessing import LabelEncoder
        from catboost import CatBoostClassifier

        # In questo stadio X è già numerico, quindi non servono cat_features
        model = CatBoostClassifier(iterations=100, random_state=42, verbose=0)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        model.fit(X, y_encoded)
        
        scores = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        plot_feature_ranking(scores, title="CatBoost Importance", save_path="plots/catboost_importance.png")
        return scores

    def relief_importances(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calcola l'importanza delle feature tramite l'algoritmo ReliefF."""
        from skrebate import ReliefF
        
        relief = ReliefF(n_neighbors=100)
        relief.fit(X.values, y.values)
        
        scores = pd.Series(relief.feature_importances_, index=X.columns).sort_values(ascending=False)
        plot_feature_ranking(scores, title="ReliefF Ranking", save_path="plots/relief_ranking.png")
        return scores

    def pca_transformation(self, X: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
        """Esegue la Feature Extraction tramite PCA."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        return pd.DataFrame(
            X_pca, 
            columns=[f"PCA_Comp_{i+1}" for i in range(n_components)],
            index=X.index
        )

    def rfe_selection(self, X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 30) -> pd.Series:
        """Seleziona le feature tramite Recursive Feature Elimination (RFE)."""
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestClassifier
        
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=5)
        selector = selector.fit(X, y)
        
        # Creiamo un ranking invertito (1 è il migliore, quindi lo trasformiamo per coerenza con gli altri metodi)
        ranking = pd.Series(1.0 / selector.ranking_, index=X.columns).sort_values(ascending=False)
        return ranking

    def sfs_selection(self, X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 15) -> pd.Series:
        """Seleziona le feature tramite Sequential Feature Selection (SFS)."""
        from sklearn.feature_selection import SequentialFeatureSelector
        from sklearn.linear_model import LogisticRegression
        
        estimator = LogisticRegression(max_iter=500)
        sfs = SequentialFeatureSelector(estimator, n_features_to_select=n_features_to_select, direction='forward', n_jobs=-1)
        sfs.fit(X, y)
        
        # SFS restituisce una maschera booleana
        scores = pd.Series(sfs.get_support().astype(float), index=X.columns).sort_values(ascending=False)
        return scores

    @staticmethod
    def autoencoder_extraction(X: pd.DataFrame, encoding_dim: int = 16, epochs: int = 10):
        """Esegue la Feature Extraction tramite Autoencoder."""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense
        from tensorflow.keras.optimizers import Adam
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        input_dim = X_scaled.shape[1]

        input_layer = Input(shape=(input_dim,))
        encoded = Dense(64, activation="relu")(input_layer)
        encoded = Dense(encoding_dim, activation="relu")(encoded)
        decoded = Dense(64, activation="relu")(encoded)
        decoded = Dense(input_dim, activation="sigmoid")(decoded)

        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)

        autoencoder.compile(optimizer=Adam(), loss="mse")
        autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=32, verbose=0)

        X_latent = encoder.predict(X_scaled)
        return pd.DataFrame(
            X_latent, 
            columns=[f"AE_Feature_{i+1}" for i in range(encoding_dim)],
            index=X.index
        )
    