"""
REGRESIÓN LINEAL

Modelo:
    y = m * x + b

Objetivo: minimizar el Error Cuadrático Medio

"""

from typing import List
import math
import matplotlib.pyplot as plt


class LinearRegressionScratch:
    def __init__(self, lr: float = 1e-3, epochs: int = 3000):
        self.lr = lr
        self.epochs = epochs

        self.m = 0.0
        self.b = 0.0

        self.loss_history: List[float] = []

        self._x_mean = 0.0
        self._x_std = 1.0
        self._y_mean = 0.0

        self._m_s = 0.0
        self._b_s = 0.0

    @staticmethod
    def _mse(y_true: List[float], y_pred: List[float]) -> float:
        """
          Monitorear la convergencia
        """
        n = len(y_true)
        return sum((float(yt) - float(yp)) ** 2 for yt, yp in zip(y_true, y_pred)) / n

    def _standardize_x(self, X: List[float]) -> List[float]:
        """
          Estandariza la variable independiente X.
        """
        return [(xi - self._x_mean) / self._x_std for xi in X]

    def _destandardize_params(self):
        """
          Desestandariza los parámetros m y b.
        """
        m_orig = self._m_s / self._x_std
        b_orig = (self._b_s - self._m_s * self._x_mean /
                  self._x_std) + self._y_mean
        self.m = m_orig
        self.b = b_orig

    def fit(self, X: List[float], y: List[float], verbose_every: int = 0) -> None:
        """
          Entrena el modelo con descenso de gradiente en el espacio estandarizado.
        """
        assert len(X) == len(y) and len(
            X) > 0
        n = len(X)
        self.loss_history.clear()

        self._x_mean = sum(X) / n
        x_var = sum((xi - self._x_mean) ** 2 for xi in X) / n
        self._x_std = math.sqrt(x_var) if x_var > 0 else 1.0

        self._y_mean = sum(y) / n

        Z = self._standardize_x(X)
        y_center = [yi - self._y_mean for yi in y]

        self._m_s = 0.0
        self._b_s = 0.0

        for epoch in range(self.epochs):
            y_pred_c = [self._m_s * zi + self._b_s for zi in Z]  # y' pred

            dm_num = sum((yci - ypci) * zi for zi, yci,
                         ypci in zip(Z, y_center, y_pred_c))
            db_num = sum((yci - ypci) for yci, ypci in zip(y_center, y_pred_c))

            dm = (-2.0 / n) * dm_num
            db = (-2.0 / n) * db_num

            max_abs = 1e6
            if dm > max_abs:
                dm = max_abs
            if dm < -max_abs:
                dm = -max_abs
            if db > max_abs:
                db = max_abs
            if db < -max_abs:
                db = -max_abs

            self._m_s -= self.lr * dm
            self._b_s -= self.lr * db

            y_pred = [self._m_s * zi + self._b_s + self._y_mean for zi in Z]
            loss = self._mse(y, y_pred)
            self.loss_history.append(loss)

            if verbose_every and epoch % verbose_every == 0:
                m_tmp = self._m_s / self._x_std
                b_tmp = (self._b_s - self._m_s * self._x_mean /
                         self._x_std) + self._y_mean
                print(
                    f"[epoch {epoch:4d}] MSE = {loss:.6f} | m={m_tmp:.6f} b={b_tmp:.6f}")

        self._destandardize_params()

    def predict(self, X: List[float]) -> List[float]:
        """
          Realiza predicciones para nuevos datos.
        """
        return [self.m * xi + self.b for xi in X]

    def r2_score(self, X, y):
        """
          Calcula el coeficiente de determinación R^2.
        """
        y_pred = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))
        return 1 - (ss_res / ss_tot if ss_tot != 0 else float('nan'))

    def plot_loss(self, save_path: str | None = None) -> None:
        """
          Grafica la evolución de la pérdida durante el entrenamiento.
        """
        if not self.loss_history:
            print("No hay historial de pérdida. Entrena primero con fit().")
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history)
        plt.xlabel("Épocas")
        plt.ylabel("MSE")
        plt.title("Curva de entrenamiento")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=160, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_regression(self, X: List[float], y: List[float], save_path: str | None = None) -> None:
        """
          Grafica la regresión lineal ajustada a los datos.
        """
        if not X or not y:
            print("Datos vacíos.")
            return
        pairs = sorted(zip(X, y), key=lambda p: p[0])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        y_line = self.predict(xs)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(X, y, label="Datos", alpha=0.9)
        plt.plot(xs, y_line, label=f"y = {self.m:.3f}x + {self.b:.3f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Regresión lineal (univariada)")
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=160, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
