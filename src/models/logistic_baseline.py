from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def get_supervised_models():
    """Returns supervised baseline models defined in Phase 3."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500,
            random_state=42, early_stopping=True
        ),
    }
