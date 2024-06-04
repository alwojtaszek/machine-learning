from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data():
    breast_cancer = load_breast_cancer()
    data = breast_cancer['data']
    labels = breast_cancer['target']

    # Standardize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data, labels


def train_and_evaluate_model(X, y, n_components):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Cross-validation score
    cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
    print(f'Cross-validation score with {n_components} components: {cv_score:.4f}')

    # Predict and evaluate on the test set
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f'Confusion matrix with {n_components} components:')
    print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')
    print('---------------------------------------')

    return cv_score, tn, fp, fn, tp


def main():
    data, labels = load_and_preprocess_data()
    components = [5, 10, 15, 20, 25, 30]
    results = {}

    for n in components:
        pca = PCA(n_components=n)
        data_pca = pca.fit_transform(data)
        cv_score, tn, fp, fn, tp = train_and_evaluate_model(data_pca, labels, n)
        results[n] = {'cv_score': cv_score, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    return results


if __name__ == "__main__":
    results = main()