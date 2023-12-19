from typing import Tuple, Callable, Union, List
import json
from scipy.stats import  logistic
import matplotlib.pyplot as plt
import numpy as np
import random

"""
Пусть есть два события связаны соотношением:
P{y=1|X} = f(z) (1)
P{y=0|X} = 1 - f(z) (2)
z = b + (X,T)
f(z)  = 1 / (1 + exp{-z})
d/dz f(z)  = z' * exp{-z} / (1 + exp{-z})^2 = z' * (1 - 1/f(z))/f(z)^2 = 
           = z' * ((1 - f(z)) * f(z)) (3)

Тогда соотношения (1) и (2):
P{y=1|X} = f(b + (X,T)) (4)
P{y=0|X} = 1 - f(b + (X,T)) (5)
Вероятность y при условии Х
P{y|X} = f(b + (X,T))^y*(1 - f(b + (X,T)))^(y-1) (6)
Условие максимального правдоподобия:
{b,T} = argmax(П P{y_i|X_i}) = argmax(Σ ln(P{y_i|X_i}))
argmax(Σ ln(P{y_i|X_i})) = Σ y_i * ln(f(b + (X,T))) + (1-y_i) * (ln(1 - f(b + (X,T))))
требуеся найти производные для:
d/db argmax(Σ ln(P{y_i|X_i}))
d/t_j argmax(Σ ln(P{y_i|X_i})), где t_j элемент вектора T
Для этого распишем необходимые нам формулы:
d/dx ln(f(x)) = af'(x)/f(x)

d/db   ln(f(b + (X,T))) =       f'(b + (X,T))/f(b + (X,T)) =        1 - f(b + (X,T) 
d/dx_j ln(f(b + (X,T))) = d/dx_j f(b + (X,T))/f(b + (X,T)) = x_j * (1 - f(b + (X,T))
"""

_debug_mode = True
_accuracy = 1e-6
epsilon = 1e-15
Vector2Int = Tuple[int, int]
Vector2 = Tuple[float, float]
Section = Tuple[Vector2, Vector2]
EmptyArray = np.ndarray([])


def march_squares_2d(field: Callable[[float, float], float],
                     min_bound: Vector2 = (-5.0, -5.0),
                     max_bound: Vector2 = (5.0, 5.0),
                     march_resolution: Vector2Int = (128, 128),
                     threshold: float = 0.5) -> List[Section]:
    """
    Эта функция рисует неявнозаданную функцию вида f(x,y) = 0. Есть аналог в matplotlib
    :param field:
    :param min_bound:
    :param max_bound:
    :param march_resolution:
    :param threshold:
    :return:
    """

    def lin_interp(_a: float, _b: float, t) -> float:
        return _a + (_b - _a) * t

    rows, cols = max(march_resolution[1], 3), max(march_resolution[0], 3)
    cols_ = cols - 1
    rows_ = cols - 1
    dx = (max_bound[0] - min_bound[0]) / cols_
    dy = (max_bound[1] - min_bound[1]) / rows_
    row: float
    col: float
    state: int
    p1: float
    p2: float
    p3: float
    p4: float
    t_val: float
    d_t: float
    shape: List[Section] = []
    for i in range(cols_ * rows_):
        state = 0
        row = (i // cols_) * dy + min_bound[1]
        col = (i % cols_) * dx + min_bound[0]

        a_val = field(col, row)
        b_val = field(col + dx, row)
        c_val = field(col + dx, row + dy)
        d_val = field(col, row + dy)

        state += 8 if a_val >= threshold else 0
        state += 4 if b_val >= threshold else 0
        state += 2 if c_val >= threshold else 0
        state += 1 if d_val >= threshold else 0

        if state == 0 or state == 15:
            continue
        # без интерполяции
        # a = (col + dx * 0.5, row           )
        # b = (col + dx,       row + dy * 0.5)
        # c = (col + dx * 0.5, row + dy      )
        # d = (col,            row + dy * 0.5)

        d_t = b_val - a_val
        if np.abs(d_t) < _accuracy:
            a = (lin_interp(col, col + dx, np.sign(threshold - a_val)), row)
        else:
            t_val = (threshold - a_val) / d_t
            a = (lin_interp(col, col + dx, t_val), row)

        d_t = c_val - b_val
        if np.abs(d_t) < _accuracy:
            b = (col + dx, lin_interp(row, row + dy, np.sign(threshold - b_val)))
        else:
            t_val = (threshold - b_val) / d_t
            b = (col + dx, lin_interp(row, row + dy, t_val))

        d_t = c_val - d_val
        if np.abs(d_t) < _accuracy:
            c = (lin_interp(col, col + dx, np.sign(threshold - d_val)), row + dy)
        else:
            t_val = (threshold - d_val) / d_t
            c = (lin_interp(col, col + dx, t_val), row + dy)

        d_t = d_val - a_val
        if np.abs(d_t) < _accuracy:
            d = (col, lin_interp(row, row + dy, np.sign(threshold - a_val)))
        else:
            t_val = (threshold - a_val) / d_t
            d = (col, lin_interp(row, row + dy, t_val))

        while True:
            if state == 1:
                shape.append((c, d))
                break
            if state == 2:
                shape.append((b, c))
                break
            if state == 3:
                shape.append((b, d))
                break
            if state == 4:
                shape.append((a, b))
                break
            if state == 5:
                shape.append((a, d))
                shape.append((b, c))
                break
            if state == 6:
                shape.append((a, c))
                break
            if state == 7:
                shape.append((a, d))
                break
            if state == 8:
                shape.append((a, d))
                break
            if state == 9:
                shape.append((a, c))
                break
            if state == 10:
                shape.append((a, b))
                shape.append((c, d))
                break
            if state == 11:
                shape.append((a, b))
                break
            if state == 12:
                shape.append((b, d))
                break
            if state == 13:
                shape.append((b, c))
                break
            if state == 14:
                shape.append((c, d))
                break
            break
    return shape


def rand_in_range(rand_range: Union[float, Tuple[float, float]] = 1.0) -> float:
    if isinstance(rand_range, float):
        return random.uniform(-0.5 * rand_range, 0.5 * rand_range)
    if isinstance(rand_range, tuple):
        return random.uniform(rand_range[0], rand_range[1])
    return random.uniform(-0.5, 0.5)


def ellipsoid(x: float, y: float, params: Tuple[float, float, float, float, float]) -> float:
    """
    уравнение эллипсойда
    :param x: координата по х
    :param y: координата по н
    :param params: расстояние от точки (x,y) до эллипсойда f(x, y), f(x, y) = 0 <- точка принадлежит эллипсойду
    :return:
    """
    return x * params[0] + y * params[1] + x * y * params[2] + x * x * params[3] + y * y * params[4] - 1


def log_reg_ellipsoid_test_data(params: Tuple[float, float, float, float, float],
                                arg_range: float = 5.0, rand_range: float = 1.0,
                                n_points: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генератор тестовых данных для логистической регрессии. что бы понять что тут просходит, просто  _debug_mode = True
    :param params:
    :param arg_range:
    :param rand_range:
    :param n_points:
    :return:
    """
    if _debug_mode:
        print(f"logistic regression f(x,y) = {params[0]:1.3}x + {params[1]:1.3}y + {params[2]:1.3}xy +"
              f" {params[3]:1.3}x^2 + {params[4]:1.3}y^2 - 1,\n"
              f" arg_range =  [{-arg_range * 0.5:1.3}, {arg_range * 0.5:1.3}],\n"
              f" rand_range = [{-rand_range * 0.5:1.3}, {rand_range * 0.5:1.3}]")
    features = np.zeros((n_points, 5), dtype=float)
    features[:, 0] = np.array([rand_in_range(arg_range) for _ in range(n_points)])
    features[:, 1] = np.array([rand_in_range(arg_range) for _ in range(n_points)])
    features[:, 2] = features[:, 0] * features[:, 1]
    features[:, 3] = features[:, 0] * features[:, 0]
    features[:, 4] = features[:, 1] * features[:, 1]
    groups = np.array(
        [np.sign(ellipsoid(features[i, 0], features[i, 1], params)) * 0.5 + 0.5 for i in range(n_points)])
    return features, groups


def log_reg_test_data(k: float = -1.5, b: float = 0.1, arg_range: float = 1.0,
                      rand_range: float = 0.0, n_points: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генератор тестовых данных для логистической регрессии. что бы понять что тут просходит, просто  _debug_mode = True
    :param k:
    :param b:
    :param arg_range:
    :param rand_range:
    :param n_points:
    :return:
    """
    if _debug_mode:
        print(f"logistic regression test data b = {b:1.3}, k = {k:1.3},\n"
              f" arg_range = [{-arg_range * 0.5:1.3}, {arg_range * 0.5:1.3}],\n"
              f" rand_range = [{-rand_range * 0.5:1.3}, {rand_range * 0.5:1.3}]")
    features = np.zeros((n_points, 2), dtype=float)
    features[:, 0] = np.array([rand_in_range(arg_range) for _ in range(n_points)])
    features[:, 1] = np.array([rand_in_range(arg_range) for _ in range(n_points)])
    groups = np.array(
        [1 if features[i, 0] * k + b > features[i, 1] + rand_in_range(rand_range) else 0.0 for i in range(n_points)])
    return features, groups


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Вычисляет сигмоид-функцию для входного массива x.

    :param x: Входной массив
    :return: Результат сигмоид-функции для каждого элемента x
    """
    return logistic.cdf(x)
    #return 1.0 / (1.0 + np.exp(-x))


def loss(groups_probs, groups):
    """
    Функция потерь
    :param groups_probs: Вероятности принадлежности к классу 1 (предсказанные моделью)
    :param groups: Фактические метки классов (1 или 0)
    :return: Значение функции потерь
    """
    epsilon = 1e-15
    return -(groups * np.log(groups_probs + epsilon) - (1.0 - groups) * np.log(1.0 - groups_probs)).mean()


def draw_logistic_data(features: np.ndarray, groups: np.ndarray, theta: np.ndarray = None) -> None:
    """
    Рисует результат вычисления логистической регрессии
    :param features:
    :param groups:
    :param theta:
    :return:
    """
    [plt.plot(features[i, 0], features[i, 1], '+b') if groups[i] == 0
     else plt.plot(features[i, 0], features[i, 1], '*r') for i in range(features.shape[0] // 2)]

    if theta is None:
        plt.show()
        return

    b = theta[0] / np.abs(theta[2])
    k = theta[1] / np.abs(theta[2])

    x_0, x_1 = features[:, 0].min(), features[:, 0].max()
    y_0, y_1 = features[:, 1].min(), features[:, 1].max()

    y_1 = (y_1 - b) / k
    y_0 = (y_0 - b) / k

    if y_0 < y_1:
        x_1 = min(x_1, y_1)
        x_0 = max(x_0, y_0)
    else:
        x_1 = min(x_1, y_0)
        x_0 = max(x_0, y_1)

    x = [x_0, x_1]
    y = [b + x_0 * k, b + x_1 * k]
    plt.plot(x, y, 'k')
    plt.show()


class LogisticRegression:
    def __init__(self, learning_rate: float = 1.0,
                 max_iters: int = 1000, accuracy: float = 1e-2):
        # максимальное количество шагов градиентным спуском
        self._max_train_iters: int = 0
        # длина шага вдоль направления градиента
        self._learning_rate: float = 0
        # точность к которой мы стремимся
        self._learning_accuracy: float = 0
        # колическто признаков одной группы
        self._group_features_count = 0
        # параметры тетта (подробное описание в pdf файле)
        self._thetas: Union[np.ndarray, None] = None
        # текущее знаение функции потерь
        self._losses: float = 0.0

        self.max_train_iters = max_iters
        self.learning_rate = learning_rate
        self.learning_accuracy = accuracy

    def __str__(self):
        """
        JSON совместимая строка, состоящая из:
        group_features_count - целое число;,
        max_train_iters - целое число;
        learning_rate - число с плавающей точкой;
        learning_accuracy - число с плавающей точкой;
        thetas - массив значений с плавающей точкой;
        losses - число с плавающей точкой.
        :return:
        """
        # data = {
        #     "group_features_count": self._group_features_count,
        #     "max_train_iters": self._max_train_iters,
        #     "learning_rate": self._learning_rate,
        #     "learning_accuracy": self._learning_accuracy,
        #     "thetas": self._thetas.tolist() if self._thetas is not None else [],
        #     "losses": self._losses
        # }
        return f"{{\n\tgroup_features_count: {self._group_features_count}, " \
               f"\n\tmax_train_iters: {self._max_train_iters}, " \
               f"\n\tlearning_rate: {self._learning_rate}, " \
               f"\n\tlearning_accuracy: {self._learning_accuracy}, " \
               f"\n\tthetas: {self._thetas}, " \
               f"\n\tlosses: {self._losses}\n}}"
        #return json.dumps(data)

    @property
    def group_features_count(self) -> int:
        return self._group_features_count

    @property
    def max_train_iters(self) -> int:
        return self._max_train_iters

    @max_train_iters.setter
    def max_train_iters(self, value: int) -> None:
        # проверка типа
        # проверка диапаона от до, но не меньше, например, 100
        if isinstance(value, int) and value >= 100:
            self._max_train_iters = value
        else:
            raise ValueError("max_train_iters должно быть целым числом большим или равным 100")

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        # проверка типа
        # проверка диапаона от до, но не меньше, например, 0.01 и не больше 1.0
        if isinstance(value, (float, int)) and 0.01 <= value <= 1.0:
            self._learning_rate = float(value)
        else:
            raise ValueError("learning_rate должно быть большим или равным 0.01, но меньшим или равным 1.0")

    @property
    def learning_accuracy(self) -> float:
        return self._learning_accuracy

    @learning_accuracy.setter
    def learning_accuracy(self, value: float) -> None:
        # проверка типа
        # проверка диапаона от до, но не меньше, например, 0.01 и не больше 1.0
        if isinstance(value, (float, int)) and 0.01 <= value <= 1.0:
            self._learning_accuracy = float(value)

    @property
    def thetas(self) -> np.ndarray:
        if self._thetas is not None:
            return self._thetas
        else:
            return np.array([])

    @property
    def losses(self) -> float:
        return self._losses

    def predict(self, features: np.ndarray) -> np.ndarray:
        # проверка размерности -  количество принаков группы == количество элементов в толбце
        n_samples, n_features = features.shape
        assert n_features == self._thetas.size - 1
        return sigmoid(np.dot(features, self._thetas[1:]) + self._thetas[0])

    def train(self, features: np.ndarray, groups: np.ndarray) -> None:
        """
        :param features: - признаки групп, записанные в виде столбцов
        :param groups: - вектор столбец принадлежности групп (0-первая группа, 1-вторая)
        :return:
        """
        # проверка размерности -  количество принаков группы == количество элементов в толбце
        # реализация градиентного спуска для обучения логистической регрессии.
        # формула thetas(i) = thetas(i - 1) - learning_rate * (X^T * sigmoid(X *  thetas(i - 1)) - groups)
        n_samples, n_features = features.shape
        assert n_samples == groups.size

        self._group_features_count = n_features
        self._thetas = np.zeros(n_features + 1)
        X = np.hstack([np.ones(shape=(n_samples, 1)), features])

        for _ in range(self._max_train_iters):
            logits = sigmoid(np.dot(X, self._thetas))
            self._thetas -= self._learning_rate * np.dot(X.T, logits - groups)
            # not important
            loss_v = loss(groups, logits)
            #print(f"loss_v = {loss_v}")
            #loss1 = np.multiply(np.log(logits + 1e-15), groups) + np.multiply((1 - groups), np.log(1 - logits + 1e-15))
            #print(f"loss1 = {loss1}")
            cost = -np.sum(loss_v) / groups.size
            loss = np.multiply(np.log(logits + 1e-15), groups) + np.multiply((1 - groups), np.log(1 - logits + 1e-15))
            cost = -np.sum(loss) / groups.size
            self._losses = cost
            if (self._losses <= self._learning_accuracy):
                #print(f"\nself._losses = {self._losses} и self._learning_accuracy = {self._learning_accuracy}\n")
                break

        if _debug_mode:
            print(f"Полученные значения весов после обучения: {self._thetas}")
            print(f"Значения точности и общей потери: {self._learning_accuracy, self._losses}\n")


def lin_reg_test():
    features, group = log_reg_test_data()
    lg = LogisticRegression()
    lg.train(features, group)
    draw_logistic_data(features, group, lg.thetas)


def non_lin_reg_test():
    features, group = log_reg_ellipsoid_test_data((0.08, -0.08, 1.6, 1.0, 1.0))
    lg = LogisticRegression()
    lg.train(features, group)
    print(f"Обученные параметры логистической регрессии:\n {lg}")

    thetas = lg.thetas

    def _ellipsoid(x: float, y: float) -> float:
        return thetas[0] + x * thetas[1] + y * thetas[2] + x * y * thetas[3] + x * x * thetas[4] + y * y * thetas[5]

    sections = march_squares_2d(_ellipsoid)

    for arc in sections:
        p_0, p_1 = arc
        plt.plot([p_0[0], p_1[0]], [p_0[1], p_1[1]], 'k')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    print(f"\nНормализованные значения параметров тетта логистической регрессии: {lg.thetas / np.abs(lg.thetas[0])}")
    draw_logistic_data(features, group)


if __name__ == "__main__":
    lin_reg_test()
    non_lin_reg_test()