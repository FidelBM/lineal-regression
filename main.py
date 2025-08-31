import csv
from regression import LinearRegressionScratch


def load_data(path: str):
    """
    Carga CSV 
    """
    X, y = [], []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            X.append(float(row[0]))
            y.append(float(row[5]))
    return X, y


if __name__ == "__main__":
    X, y = load_data("data/bank.csv")

    model = LinearRegressionScratch(lr=0.01, epochs=1500)
    model.fit(X, y, verbose_every=200)

    r2 = model.r2_score(X, y)
    print(f"R^2 = {r2:.4f}")

    test_values = [2, 4, 6]
    preds = model.predict(test_values)

    print("\n--- Resultados ---")
    print(f"Recta aprendida: y = {model.m:.4f}x + {model.b:.4f}")
    for xv, pv in zip(test_values, preds):
        print(f"x={xv:>5} -> y_pred={pv:.4f}")

    model.plot_loss()
    model.plot_regression(X, y)
