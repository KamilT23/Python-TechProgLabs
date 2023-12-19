import clustering_utils
from clustering_utils import gaussian_cluster, draw_clusters, distance, gauss_core
from typing import Union, List, Tuple
import numpy as np


def manhattan_distance(left, right) -> float:
    return sum(abs(val1 - val2) for val1, val2 in zip(right, left))

class MShift:
    _MINIMAL_DISTANCE_THRESHOLD = 1e-6

    def __init__(self):
        """
        Метод среднего сдвига.
        Этапы алгоритма:
        1. Копируем переданные данные для кластеризации "data" в "data_shifted"
        2. Для каждой точки из набора данных "data" с индексом "point_index" считаем среднее значение вокруг нее.
        3. Полученное значение на среднего сравниваем со значением под индексом "point_index" из "data_shifted"
           Если расстояние между точками меньше, чем некоторый порог "distance_threshold", то говорим, что смещать
           точку далее нет смысла и помечаем ее, как неподвижную.
           Для новой неподвижной точки пытаемся найти ближайший к ней центр кластера и добавить индекс этой точки
           к индексам точек кластера. Кластер считается ближайшим, если расстояние между точкой и центроидом
           меньше, чем "window_size".
           Если такового нет, то точка считается новым центром кластера.
        4. Пункты 2-3 повторяются до тех пор, пока все точки не будут помечены, как неподвижные.
        """
        """
        Количество кластеров, которые ожидаем обнаружить.
        """
        self._n_clusters: int = -1
        """
        Данные для кластеризации. Поле инициализируется при вызове метода fit.
        Например, если данные представляют собой координаты на плоскости,
        каждая отельная строка - это точка этой плоскости.
        """
        self._data: Union[np.ndarray, None] = None
        """
        Центры кластеров на текущем этапе кластеризации.
        """
        self._clusters_centers: Union[List[np.ndarray], None] = None
        """
        Список индексов строк из "_data", которые соответствуют определённому кластеру.
        Список списков индексов.
        """
        self._clusters_points_indices: Union[List[List[int]], None] = None
        """
        Расстояние между центроидом кластера на текущем шаге и предыдущем при котором завершается кластеризация.
        """
        self._distance_threshold: float = 1e-3
        """
        Ширина ядра функции усреднения.
        """
        self._window_size: float = 0.15

    @property
    def window_size(self) -> float:
        """
        Просто геттер для ширины ядра функции усреднения ("_window_size").
        """
        return self._window_size

    @window_size.setter
    def window_size(self, value: float) -> None:
        """
        Сеттер для ширины ядра функции усреднения ("_window_size").
        1. Должен осуществлять проверку типа.
        2. Проверку на не отрицательность.
        """
        assert isinstance(value, float)
        assert (value >= 0)
        self._window_size = value

    @property
    def distance_threshold(self) -> float:
        """
        Просто геттер для "_distance_threshold".
        """
        return self._distance_threshold

    @distance_threshold.setter
    def distance_threshold(self, value: float) -> None:
        """
        Сеттер для "_distance_threshold".
        1. Должен осуществлять проверку типа.
        2. Проверку на не отрицательность.
        """
        assert isinstance(value, float)
        assert isinstance(value, float)
        self._distance_threshold = max(value, MShift._MINIMAL_DISTANCE_THRESHOLD)

    @property
    def n_clusters(self) -> int:
        """
        Геттер для числа кластеров, которые обнаружили.
        """
        if self._n_clusters is None:
            return len(self._clusters_centers)
        else:
            return 0

    @property
    def n_samples(self) -> int:
        """
        Количество записей в массиве данных. Например, количество {x, y} координат на плоскости.
        """
        return self._data.shape[0]

    @property
    def n_features(self) -> int:
        """
        Количество особенностей каждой записи в массив денных. Например,
        две координаты "x" и "y" в случе точек на плоскости.
        """
        return self._data.shape[1]

    @property
    def clusters(self) -> List[np.ndarray]:
        """
        Создаёт список из np.ndarray. Каждый такой массив - это все точки определённого кластера.
        Индексы точек соответствующих кластеру хранятся в "_clusters_points_indices"
        """
        if self._clusters_points_indices is not None:
            return [self._data[indices] for indices in self._clusters_points_indices]
        else:
            return []

    def _clear_current_clusters(self) -> None:
        """
        Очищает центры кластеров на текущем этапе кластеризации.
        Очищает список индексов строк из "_data", которые соответствуют определённому кластеру.
        Реализует "ленивую" инициализацию полей "_clusters" и "_clusters_centers".
        """
        if self._clusters_points_indices is None:
            self._clusters_centers = []
            self._clusters_points_indices = []
        self._clusters_centers.clear()
        self._clusters_points_indices.clear()

    def _shift_cluster_point(self, point: np.ndarray) -> np.ndarray:
        """
        Функция, которая считает средне-взвешенное (если, например, используется Гауссово ядро) внутри круглого окна
        с радиусом "window_size" вокруг точки point.
        Возвращает массив равный по размеру "point".
        """
        dists = np.linalg.norm(self._data - point, axis=1)
        weights = clustering_utils.gauss_core(dists, self._window_size)

        shift = np.sum(self._data.T * weights, axis=1)
        scale_factor = np.sum(weights)

        return shift / scale_factor if scale_factor != 0.0 else point.copy()

    def _update_clusters_centers(self, sample_index, sample: np.ndarray):
        """
        Функция ищет ближайший центр кластера для точки "sample".
        Если не находит, то считает, что "sample" - новый центр кластера.
        Если находит, то добавляет к индексам точек кластера "sample_index"
        """
        if self._n_clusters == 0:
            self._clusters_centers.append(sample)
            self._clusters_points_indices.append([sample_index])
        closest_cluster_center_index, dist = self._get_closest_cluster_center(sample)
        # print(f"индекс кластера  {closest_cluster_center_index}")
        if dist < self._window_size:
            self._clusters_points_indices[closest_cluster_center_index].append(sample_index)
        else:
            self._clusters_centers.append(sample)
            self._clusters_points_indices.append([sample_index])

    def _shift_cluster_points(self, max_iters: int = 1000) -> None:
        """
        Выполняет итеративный сдвиг всех точек к их среднему значению.
        Т.е. для каждой точки вызывается функция _shift_cluster_point()
        Выполняется до тех пор, пока все точки не будут помечены, как неподвижные.
        """
        shifted_points = np.array(self._data)
        frozen_points = set()

        while len(frozen_points) < self.n_samples:
            for sample_index, sample in enumerate(shifted_points):
                if sample_index not in frozen_points:
                    shifted_sample = self._shift_cluster_point(sample)
                    dist = manhattan_distance(shifted_sample, sample)
                    shifted_points[sample_index] = shifted_sample

                    if dist <= self._distance_threshold:
                        frozen_points.add(sample_index)
                        self._update_clusters_centers(sample_index, shifted_sample)

    def _get_closest_cluster_center(self, sample: np.ndarray) -> Tuple[int, float]:
        """
        Определяет ближайший центр кластера для точки из переданного набора данных и расстояние до него.
        Hint: для ускорения кода используйте min с генератором.
        """
        distances = [manhattan_distance(center, sample) for center in self._clusters_centers]
        return min(enumerate(distances), key=lambda x: x[1], default=(-1, float('inf')))

    def fit(self, data: np.ndarray) -> None:
        """
        Выполняет кластеризацию данных в "data".
        1. Необходима проверка, что "data" - экземпляр класса "np.ndarray".
        2. Необходима проверка, что "data" - двумерный массив.
        Этапы работы метода:
        # 1. Проверки передаваемых аргументов
        # 2. Присваивание аргументов внутренним полям класса.
        # 3. Сдвиг точек в направлении средних значений вокруг них ("_shift_cluster_points").
        """
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2

        self._data = data
        self._clear_current_clusters()
        self._shift_cluster_points()

    def show(self):
        if self._clusters_points_indices is not None:
            draw_clusters(self.clusters, cluster_centers=self._clusters_centers, title="Mean shift clustering")
        else:
            print("Clusters not initialized. Run the fit method first.")


def separated_clusters():
    """
    Пример с пятью разрозненными распределениями точек на плоскости.
    """
    m_means = MShift()
    clusters_data = np.vstack((gaussian_cluster(cx=0.5, n_points=1024),
                               gaussian_cluster(cx=1.0, n_points=1024),
                               gaussian_cluster(cx=1.5, n_points=1024),
                               gaussian_cluster(cx=2.0, n_points=1024),
                               gaussian_cluster(cx=2.5, n_points=1024)))
    m_means.fit(clusters_data)
    m_means.show()


def merged_clusters():
    """
    Пример с кластеризацией пятна.
    """
    m_means = MShift()
    m_means.fit(gaussian_cluster(n_points=1024))
    m_means.show()


if __name__ == "__main__":
    """
    Сюрприз-сюрприз! Вызов функций "merged_clusters" и "separated_clusters".
    """
    merged_clusters()
    separated_clusters()
