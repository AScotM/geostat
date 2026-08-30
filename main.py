import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree


@dataclass
class DataPoint:
    x: float
    y: float
    value: float


@dataclass
class VariogramModel:
    nugget: float
    sill: float
    range_param: float
    model_type: str


class SimpleGeostat:

    def __init__(self, random_seed: Optional[int] = None):
        self.data: List[DataPoint] = []
        self.kdtree: Optional[cKDTree] = None
        self.data_array: Optional[np.ndarray] = None
        self.random_seed = random_seed

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def load_csv(
        self,
        filename: str,
        x_col: int = 0,
        y_col: int = 1,
        val_col: int = 2,
        has_header: bool = True,
        reset: bool = False,
    ) -> None:
        if x_col < 0 or y_col < 0 or val_col < 0:
            raise ValueError("Column indexes must be non-negative")

        if reset:
            self.clear_data()

        try:
            with open(filename, "r", encoding="utf-8", newline="") as file:
                reader = csv.reader(file)

                if has_header:
                    try:
                        next(reader)
                    except StopIteration:
                        pass

                for row in reader:
                    try:
                        x = float(row[x_col])
                        y = float(row[y_col])
                        value = float(row[val_col])

                        if not all(math.isfinite(v) for v in (x, y, value)):
                            continue

                        self.data.append(
                            DataPoint(
                                x=x,
                                y=y,
                                value=value,
                            )
                        )

                    except (ValueError, IndexError):
                        continue

        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File {filename} not found") from exc
        except PermissionError as exc:
            raise PermissionError(
                f"Permission denied for file {filename}"
            ) from exc

        self._build_spatial_index()

    def _build_spatial_index(self) -> None:
        if not self.data:
            self.kdtree = None
            self.data_array = None
            return

        self.data_array = np.array(
            [[point.x, point.y, point.value] for point in self.data],
            dtype=float,
        )

        self.kdtree = cKDTree(self.data_array[:, :2])

    def save_csv(
        self,
        filename: str,
        predictions: List[Tuple[Tuple[float, float], float]],
    ) -> None:
        try:
            with open(filename, "w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["x", "y", "predicted"])

                for (x, y), prediction in predictions:
                    writer.writerow([x, y, prediction])

        except PermissionError as exc:
            raise PermissionError(
                f"Permission denied for file {filename}"
            ) from exc

    def clear_data(self) -> None:
        self.data = []
        self.kdtree = None
        self.data_array = None

    def create_grid(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        resolution: int,
    ) -> Generator[Tuple[float, float], None, None]:
        if not all(math.isfinite(v) for v in (xmin, xmax, ymin, ymax)):
            raise ValueError("Grid bounds must be finite")

        if xmin >= xmax:
            raise ValueError("xmin must be less than xmax")

        if ymin >= ymax:
            raise ValueError("ymin must be less than ymax")

        if not isinstance(resolution, int):
            raise TypeError("Resolution must be an integer")

        if resolution <= 0:
            raise ValueError("Resolution must be positive")

        x_step = (xmax - xmin) / resolution
        y_step = (ymax - ymin) / resolution

        for i in range(resolution + 1):
            for j in range(resolution + 1):
                x = xmin + i * x_step
                y = ymin + j * y_step
                yield x, y

    def idw(
        self,
        target_x: float,
        target_y: float,
        power: float = 2.0,
        max_points: Optional[int] = None,
        data_source: Optional[List[DataPoint]] = None,
    ) -> float:
        if not math.isfinite(target_x) or not math.isfinite(target_y):
            raise ValueError("Target coordinates must be finite")

        if not math.isfinite(power) or power <= 0:
            raise ValueError("Power must be positive and finite")

        if max_points is not None:
            if not isinstance(max_points, int):
                raise TypeError("max_points must be an integer")

            if max_points <= 0:
                raise ValueError("max_points must be positive")

        if data_source is None:
            if (
                self.kdtree is None
                or self.data_array is None
                or len(self.data_array) == 0
            ):
                return 0.0

            if max_points is not None:
                k = min(max_points, len(self.data_array))

                distances, indices = self.kdtree.query(
                    [target_x, target_y],
                    k=k,
                )

                distances = np.atleast_1d(distances).astype(float)
                indices = np.atleast_1d(indices).astype(int)

                valid = (
                    np.isfinite(distances)
                    & (indices >= 0)
                    & (indices < len(self.data_array))
                )

                distances = distances[valid]
                indices = indices[valid]

                if len(indices) == 0:
                    return 0.0

                neighbors = self.data_array[indices]

            else:
                neighbors = self.data_array

                distances = np.hypot(
                    neighbors[:, 0] - target_x,
                    neighbors[:, 1] - target_y,
                )

            zero_distance_mask = distances <= 1e-12

            if np.any(zero_distance_mask):
                return float(neighbors[zero_distance_mask][0, 2])

            weights = 1.0 / np.power(distances, power)
            total_weight = float(np.sum(weights))

            if (
                total_weight <= 0
                or not math.isfinite(total_weight)
            ):
                return float(np.mean(neighbors[:, 2]))

            weighted_sum = float(
                np.sum(weights * neighbors[:, 2])
            )

            return weighted_sum / total_weight

        if not data_source:
            return 0.0

        distances: List[Tuple[float, float]] = []

        for point in data_source:
            distance = math.hypot(
                point.x - target_x,
                point.y - target_y,
            )

            if distance <= 1e-12:
                return float(point.value)

            distances.append((distance, point.value))

        if max_points is not None:
            distances.sort(key=lambda item: item[0])
            distances = distances[:max_points]

        if not distances:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for distance, value in distances:
            weight = 1.0 / (distance ** power)
            total_weight += weight
            weighted_sum += weight * value

        if total_weight <= 0 or not math.isfinite(total_weight):
            return sum(
                point.value for point in data_source
            ) / len(data_source)

        return weighted_sum / total_weight

    def _block_shape(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        block_size: float,
    ) -> Tuple[int, int]:
        width = xmax - xmin
        height = ymax - ymin

        n_blocks_x = max(
            1,
            int(math.ceil(width / block_size)),
        )

        n_blocks_y = max(
            1,
            int(math.ceil(height / block_size)),
        )

        return n_blocks_x, n_blocks_y

    def _block_index(
        self,
        x: float,
        y: float,
        xmin: float,
        ymin: float,
        block_size: float,
        n_blocks_x: int,
        n_blocks_y: int,
    ) -> Tuple[int, int]:
        block_x = int((x - xmin) // block_size)
        block_y = int((y - ymin) // block_size)

        block_x = max(
            0,
            min(block_x, n_blocks_x - 1),
        )

        block_y = max(
            0,
            min(block_y, n_blocks_y - 1),
        )

        return block_x, block_y

    def block_average(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        block_size: float = 10.0,
    ) -> Dict[Tuple[int, int], float]:
        if not math.isfinite(block_size) or block_size <= 0:
            raise ValueError(
                "Block size must be positive and finite"
            )

        if xmin >= xmax:
            raise ValueError("xmin must be less than xmax")

        if ymin >= ymax:
            raise ValueError("ymin must be less than ymax")

        n_blocks_x, n_blocks_y = self._block_shape(
            xmin,
            xmax,
            ymin,
            ymax,
            block_size,
        )

        blocks: Dict[Tuple[int, int], List[float]] = defaultdict(list)

        for point in self.data:
            if (
                xmin <= point.x <= xmax
                and ymin <= point.y <= ymax
            ):
                block = self._block_index(
                    point.x,
                    point.y,
                    xmin,
                    ymin,
                    block_size,
                    n_blocks_x,
                    n_blocks_y,
                )

                blocks[block].append(point.value)

        return {
            block: sum(values) / len(values)
            for block, values in blocks.items()
        }

    def experimental_variogram(
        self,
        max_lag: float,
        n_bins: int = 20,
    ) -> Tuple[List[float], List[float]]:
        if not math.isfinite(max_lag) or max_lag <= 0:
            raise ValueError(
                "max_lag must be positive and finite"
            )

        if not isinstance(n_bins, int):
            raise TypeError("n_bins must be an integer")

        if n_bins <= 0:
            raise ValueError("n_bins must be positive")

        if len(self.data) < 2:
            return [], []

        points_array = np.array(
            [
                [point.x, point.y, point.value]
                for point in self.data
            ],
            dtype=float,
        )

        coords = points_array[:, :2]

        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=max_lag)

        if not pairs:
            return [], []

        bin_width = max_lag / n_bins

        semivariance_sums = np.zeros(n_bins, dtype=float)
        bin_counts = np.zeros(n_bins, dtype=int)

        for i, j in pairs:
            dx = points_array[i, 0] - points_array[j, 0]
            dy = points_array[i, 1] - points_array[j, 1]

            distance = math.hypot(dx, dy)

            value_difference = (
                points_array[i, 2]
                - points_array[j, 2]
            )

            semivariance = 0.5 * value_difference ** 2

            bin_index = min(
                int(distance / bin_width),
                n_bins - 1,
            )

            semivariance_sums[bin_index] += semivariance
            bin_counts[bin_index] += 1

        lag_centers: List[float] = []
        gamma: List[float] = []

        for i in range(n_bins):
            if bin_counts[i] == 0:
                continue

            lag_centers.append(
                (i + 0.5) * bin_width
            )

            gamma.append(
                float(
                    semivariance_sums[i]
                    / bin_counts[i]
                )
            )

        return lag_centers, gamma

    def _spherical_variogram(
        self,
        h: float,
        nugget: float,
        sill: float,
        range_param: float,
    ) -> float:
        if h <= 0:
            return 0.0

        if range_param <= 0:
            return nugget + sill

        if h < range_param:
            ratio = h / range_param

            return (
                nugget
                + sill
                * (
                    1.5 * ratio
                    - 0.5 * ratio ** 3
                )
            )

        return nugget + sill

    def _exponential_variogram(
        self,
        h: float,
        nugget: float,
        sill: float,
        range_param: float,
    ) -> float:
        if h <= 0:
            return 0.0

        if range_param <= 0:
            return nugget + sill

        return (
            nugget
            + sill
            * (
                1.0
                - math.exp(
                    -3.0 * h / range_param
                )
            )
        )

    def _gaussian_variogram(
        self,
        h: float,
        nugget: float,
        sill: float,
        range_param: float,
    ) -> float:
        if h <= 0:
            return 0.0

        if range_param <= 0:
            return nugget + sill

        ratio = h / range_param

        return (
            nugget
            + sill
            * (
                1.0
                - math.exp(
                    -3.0 * ratio ** 2
                )
            )
        )

    def _variogram_array(
        self,
        distances: np.ndarray,
        model: VariogramModel,
    ) -> np.ndarray:
        distances = np.asarray(
            distances,
            dtype=float,
        )

        result = np.zeros_like(
            distances,
            dtype=float,
        )

        positive_mask = distances > 0

        if not np.any(positive_mask):
            return result

        h = distances[positive_mask]

        range_param = max(
            model.range_param,
            1e-12,
        )

        if model.model_type == "spherical":
            ratios = h / range_param

            values = np.full(
                h.shape,
                model.nugget + model.sill,
                dtype=float,
            )

            within_range = h < range_param

            ratio = ratios[within_range]

            values[within_range] = (
                model.nugget
                + model.sill
                * (
                    1.5 * ratio
                    - 0.5 * ratio ** 3
                )
            )

        elif model.model_type == "exponential":
            values = (
                model.nugget
                + model.sill
                * (
                    1.0
                    - np.exp(
                        -3.0 * h / range_param
                    )
                )
            )

        elif model.model_type == "gaussian":
            values = (
                model.nugget
                + model.sill
                * (
                    1.0
                    - np.exp(
                        -3.0
                        * (h / range_param) ** 2
                    )
                )
            )

        else:
            raise ValueError(
                f"Unknown model type: {model.model_type}"
            )

        result[positive_mask] = values

        return result

    def fit_variogram_model(
        self,
        lags: List[float],
        gamma: List[float],
        model_type: str = "spherical",
    ) -> VariogramModel:
        valid_models = {
            "spherical",
            "exponential",
            "gaussian",
        }

        if model_type not in valid_models:
            raise ValueError(
                f"Unknown model type: {model_type}"
            )

        if len(lags) != len(gamma):
            raise ValueError(
                "lags and gamma must have equal lengths"
            )

        if len(lags) < 3:
            return VariogramModel(
                nugget=0.0,
                sill=1.0,
                range_param=10.0,
                model_type=model_type,
            )

        lags_array = np.asarray(
            lags,
            dtype=float,
        )

        gamma_array = np.asarray(
            gamma,
            dtype=float,
        )

        if not np.all(np.isfinite(lags_array)):
            raise ValueError(
                "lags must contain only finite values"
            )

        if not np.all(np.isfinite(gamma_array)):
            raise ValueError(
                "gamma must contain only finite values"
            )

        if np.any(lags_array < 0):
            raise ValueError(
                "lags cannot contain negative values"
            )

        if np.any(gamma_array < 0):
            raise ValueError(
                "gamma cannot contain negative values"
            )

        nugget_initial = max(
            0.0,
            float(gamma_array[0]),
        )

        sill_initial = max(
            0.01,
            float(gamma_array[-1]) - nugget_initial,
        )

        range_initial = max(
            0.1,
            float(np.max(lags_array)) * 0.5,
        )

        if model_type == "spherical":

            def model_func(
                h: np.ndarray,
                nugget: float,
                sill: float,
                range_param: float,
            ) -> np.ndarray:
                h = np.asarray(h, dtype=float)

                result = np.zeros_like(
                    h,
                    dtype=float,
                )

                positive = h > 0

                if not np.any(positive):
                    return result

                range_safe = max(
                    range_param,
                    1e-12,
                )

                positive_h = h[positive]
                ratios = positive_h / range_safe

                values = np.full(
                    positive_h.shape,
                    nugget + sill,
                    dtype=float,
                )

                inside = positive_h < range_safe

                ratio = ratios[inside]

                values[inside] = (
                    nugget
                    + sill
                    * (
                        1.5 * ratio
                        - 0.5 * ratio ** 3
                    )
                )

                result[positive] = values

                return result

        elif model_type == "exponential":

            def model_func(
                h: np.ndarray,
                nugget: float,
                sill: float,
                range_param: float,
            ) -> np.ndarray:
                h = np.asarray(h, dtype=float)

                result = np.zeros_like(
                    h,
                    dtype=float,
                )

                positive = h > 0

                range_safe = max(
                    range_param,
                    1e-12,
                )

                result[positive] = (
                    nugget
                    + sill
                    * (
                        1.0
                        - np.exp(
                            -3.0
                            * h[positive]
                            / range_safe
                        )
                    )
                )

                return result

        else:

            def model_func(
                h: np.ndarray,
                nugget: float,
                sill: float,
                range_param: float,
            ) -> np.ndarray:
                h = np.asarray(h, dtype=float)

                result = np.zeros_like(
                    h,
                    dtype=float,
                )

                positive = h > 0

                range_safe = max(
                    range_param,
                    1e-12,
                )

                result[positive] = (
                    nugget
                    + sill
                    * (
                        1.0
                        - np.exp(
                            -3.0
                            * (
                                h[positive]
                                / range_safe
                            ) ** 2
                        )
                    )
                )

                return result

        try:
            parameters, _ = curve_fit(
                model_func,
                lags_array,
                gamma_array,
                p0=[
                    nugget_initial,
                    sill_initial,
                    range_initial,
                ],
                bounds=(
                    [0.0, 0.0, 0.01],
                    [np.inf, np.inf, np.inf],
                ),
                maxfev=10000,
            )

            nugget, sill, range_param = parameters

            return VariogramModel(
                nugget=max(
                    0.0,
                    float(nugget),
                ),
                sill=max(
                    0.01,
                    float(sill),
                ),
                range_param=max(
                    0.1,
                    float(range_param),
                ),
                model_type=model_type,
            )

        except (
            RuntimeError,
            ValueError,
            FloatingPointError,
        ):
            return VariogramModel(
                nugget=nugget_initial,
                sill=sill_initial,
                range_param=range_initial,
                model_type=model_type,
            )

    def _variogram_value(
        self,
        h: float,
        model: VariogramModel,
    ) -> float:
        if model.model_type == "spherical":
            return self._spherical_variogram(
                h,
                model.nugget,
                model.sill,
                model.range_param,
            )

        if model.model_type == "exponential":
            return self._exponential_variogram(
                h,
                model.nugget,
                model.sill,
                model.range_param,
            )

        if model.model_type == "gaussian":
            return self._gaussian_variogram(
                h,
                model.nugget,
                model.sill,
                model.range_param,
            )

        raise ValueError(
            f"Unknown model type: {model.model_type}"
        )

    def ordinary_kriging(
        self,
        target_x: float,
        target_y: float,
        variogram_model: VariogramModel,
        max_points: int = 20,
    ) -> Tuple[float, float]:
        if not math.isfinite(target_x) or not math.isfinite(target_y):
            raise ValueError(
                "Target coordinates must be finite"
            )

        if not isinstance(max_points, int):
            raise TypeError(
                "max_points must be an integer"
            )

        if max_points <= 0:
            raise ValueError(
                "max_points must be positive"
            )

        if variogram_model.model_type not in {
            "spherical",
            "exponential",
            "gaussian",
        }:
            raise ValueError(
                f"Unknown model type: "
                f"{variogram_model.model_type}"
            )

        if (
            self.kdtree is None
            or self.data_array is None
            or len(self.data_array) == 0
        ):
            return (
                0.0,
                variogram_model.nugget
                + variogram_model.sill,
            )

        n_total = len(self.data_array)
        k = min(max_points, n_total)

        distances, indices = self.kdtree.query(
            [target_x, target_y],
            k=k,
        )

        distances = np.atleast_1d(
            distances
        ).astype(float)

        indices = np.atleast_1d(
            indices
        ).astype(int)

        valid = (
            np.isfinite(distances)
            & (indices >= 0)
            & (indices < n_total)
        )

        distances = distances[valid]
        indices = indices[valid]

        if len(indices) == 0:
            return (
                0.0,
                variogram_model.nugget
                + variogram_model.sill,
            )

        zero_distance_mask = distances <= 1e-12

        if np.any(zero_distance_mask):
            zero_index = indices[
                zero_distance_mask
            ][0]

            return (
                float(
                    self.data_array[
                        zero_index,
                        2,
                    ]
                ),
                0.0,
            )

        neighbors = self.data_array[indices]
        n_neighbors = len(neighbors)

        coordinates = neighbors[:, :2]

        coordinate_differences = (
            coordinates[:, np.newaxis, :]
            - coordinates[np.newaxis, :, :]
        )

        pairwise_distances = np.linalg.norm(
            coordinate_differences,
            axis=2,
        )

        gamma_matrix = self._variogram_array(
            pairwise_distances,
            variogram_model,
        )

        np.fill_diagonal(
            gamma_matrix,
            0.0,
        )

        matrix = np.zeros(
            (
                n_neighbors + 1,
                n_neighbors + 1,
            ),
            dtype=float,
        )

        matrix[
            :n_neighbors,
            :n_neighbors,
        ] = gamma_matrix

        matrix[
            :n_neighbors,
            n_neighbors,
        ] = 1.0

        matrix[
            n_neighbors,
            :n_neighbors,
        ] = 1.0

        vector = np.zeros(
            n_neighbors + 1,
            dtype=float,
        )

        vector[:n_neighbors] = (
            self._variogram_array(
                distances,
                variogram_model,
            )
        )

        vector[n_neighbors] = 1.0

        try:
            solution = np.linalg.solve(
                matrix,
                vector,
            )

        except np.linalg.LinAlgError:
            regularized = matrix.copy()

            regularized[
                :n_neighbors,
                :n_neighbors,
            ] += (
                np.eye(n_neighbors)
                * 1e-8
            )

            try:
                solution = np.linalg.solve(
                    regularized,
                    vector,
                )

            except np.linalg.LinAlgError:
                kriging_weights = (
                    np.ones(
                        n_neighbors,
                        dtype=float,
                    )
                    / n_neighbors
                )

                estimated_value = float(
                    np.sum(
                        kriging_weights
                        * neighbors[:, 2]
                    )
                )

                variance = (
                    variogram_model.nugget
                    + variogram_model.sill
                )

                return (
                    estimated_value,
                    float(variance),
                )

        kriging_weights = solution[
            :n_neighbors
        ]

        lagrange_multiplier = float(
            solution[n_neighbors]
        )

        estimated_value = float(
            np.sum(
                kriging_weights
                * neighbors[:, 2]
            )
        )

        variance = float(
            np.sum(
                kriging_weights
                * vector[:n_neighbors]
            )
            + lagrange_multiplier
        )

        if not math.isfinite(variance):
            variance = (
                variogram_model.nugget
                + variogram_model.sill
            )

        return (
            estimated_value,
            max(0.0, variance),
        )

    def _validate_cross_validation(
        self,
        k_folds: int,
    ) -> int:
        if len(self.data) < 2:
            raise ValueError(
                "At least two data points are required "
                "for cross-validation"
            )

        if not isinstance(k_folds, int):
            raise TypeError(
                "k_folds must be an integer"
            )

        if k_folds < 2:
            raise ValueError(
                "k_folds must be at least 2"
            )

        return min(
            k_folds,
            len(self.data),
        )

    def _shuffled_data(
        self,
    ) -> List[DataPoint]:
        shuffled_data = self.data[:]

        if self.random_seed is not None:
            rng = random.Random(
                self.random_seed
            )
            rng.shuffle(shuffled_data)
        else:
            random.shuffle(shuffled_data)

        return shuffled_data

    def _fold_ranges(
        self,
        data_length: int,
        k_folds: int,
    ) -> List[Tuple[int, int]]:
        base_size = data_length // k_folds
        remainder = data_length % k_folds

        ranges: List[Tuple[int, int]] = []
        start = 0

        for fold in range(k_folds):
            size = (
                base_size
                + (1 if fold < remainder else 0)
            )

            end = start + size
            ranges.append((start, end))
            start = end

        return ranges

    def cross_validate_idw(
        self,
        power: float = 2.0,
        k_folds: int = 5,
        max_points: Optional[int] = None,
    ) -> float:
        if not math.isfinite(power) or power <= 0:
            raise ValueError(
                "Power must be positive and finite"
            )

        if max_points is not None:
            if not isinstance(max_points, int):
                raise TypeError(
                    "max_points must be an integer"
                )

            if max_points <= 0:
                raise ValueError(
                    "max_points must be positive"
                )

        k_folds = self._validate_cross_validation(
            k_folds
        )

        shuffled_data = self._shuffled_data()

        errors: List[float] = []

        for start, end in self._fold_ranges(
            len(shuffled_data),
            k_folds,
        ):
            test_set = shuffled_data[
                start:end
            ]

            train_set = (
                shuffled_data[:start]
                + shuffled_data[end:]
            )

            if not train_set:
                continue

            for point in test_set:
                predicted = self.idw(
                    point.x,
                    point.y,
                    power=power,
                    max_points=max_points,
                    data_source=train_set,
                )

                error = (
                    point.value
                    - predicted
                ) ** 2

                errors.append(error)

        if not errors:
            raise ValueError(
                "Cross-validation produced no predictions"
            )

        return math.sqrt(
            sum(errors) / len(errors)
        )

    def cross_validate_kriging(
        self,
        variogram_model: VariogramModel,
        k_folds: int = 5,
        max_points: int = 20,
        refit_variogram: bool = False,
    ) -> float:
        if not isinstance(max_points, int):
            raise TypeError(
                "max_points must be an integer"
            )

        if max_points <= 0:
            raise ValueError(
                "max_points must be positive"
            )

        k_folds = self._validate_cross_validation(
            k_folds
        )

        shuffled_data = self._shuffled_data()

        errors: List[float] = []

        for start, end in self._fold_ranges(
            len(shuffled_data),
            k_folds,
        ):
            test_set = shuffled_data[
                start:end
            ]

            train_set = (
                shuffled_data[:start]
                + shuffled_data[end:]
            )

            if not train_set:
                continue

            temp_geo = SimpleGeostat(
                random_seed=self.random_seed
            )

            temp_geo.data = train_set[:]
            temp_geo._build_spatial_index()

            current_model = variogram_model

            if (
                refit_variogram
                and len(train_set) >= 3
            ):
                statistics = (
                    temp_geo.statistics_summary()
                )

                x_range = (
                    statistics["x_range_max"]
                    - statistics["x_range_min"]
                )

                y_range = (
                    statistics["y_range_max"]
                    - statistics["y_range_min"]
                )

                domain_size = max(
                    x_range,
                    y_range,
                )

                if domain_size > 0:
                    max_lag = (
                        domain_size * 0.5
                    )

                    lags, gamma = (
                        temp_geo.experimental_variogram(
                            max_lag=max_lag,
                            n_bins=15,
                        )
                    )

                    if (
                        len(lags) >= 3
                        and len(gamma) >= 3
                    ):
                        current_model = (
                            temp_geo.fit_variogram_model(
                                lags,
                                gamma,
                                model_type=(
                                    variogram_model
                                    .model_type
                                ),
                            )
                        )

            for point in test_set:
                predicted, _ = (
                    temp_geo.ordinary_kriging(
                        point.x,
                        point.y,
                        current_model,
                        max_points=max_points,
                    )
                )

                error = (
                    point.value
                    - predicted
                ) ** 2

                errors.append(error)

        if not errors:
            raise ValueError(
                "Cross-validation produced no predictions"
            )

        return math.sqrt(
            sum(errors) / len(errors)
        )

    def statistics_summary(
        self,
    ) -> Dict[str, float]:
        if not self.data:
            return {}

        values = np.array(
            [
                point.value
                for point in self.data
            ],
            dtype=float,
        )

        x_coordinates = np.array(
            [
                point.x
                for point in self.data
            ],
            dtype=float,
        )

        y_coordinates = np.array(
            [
                point.y
                for point in self.data
            ],
            dtype=float,
        )

        return {
            "n_points": float(len(self.data)),
            "mean": float(np.mean(values)),
            "variance": float(
                np.var(values, ddof=0)
            ),
            "std_dev": float(
                np.std(values, ddof=0)
            ),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "x_range_min": float(
                np.min(x_coordinates)
            ),
            "x_range_max": float(
                np.max(x_coordinates)
            ),
            "y_range_min": float(
                np.min(y_coordinates)
            ),
            "y_range_max": float(
                np.max(y_coordinates)
            ),
        }

    def predict_grid_idw(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        resolution: int,
        power: float = 2.0,
        max_points: Optional[int] = None,
        verbose: bool = False,
    ) -> List[
        Tuple[
            Tuple[float, float],
            float,
        ]
    ]:
        predictions: List[
            Tuple[
                Tuple[float, float],
                float,
            ]
        ] = []

        total = (resolution + 1) ** 2

        for index, (x, y) in enumerate(
            self.create_grid(
                xmin,
                xmax,
                ymin,
                ymax,
                resolution,
            )
        ):
            if (
                verbose
                and index
                % max(
                    1,
                    total // 10,
                )
                == 0
            ):
                print(
                    f"Progress: {index}/{total}"
                )

            prediction = self.idw(
                x,
                y,
                power=power,
                max_points=max_points,
            )

            predictions.append(
                (
                    (x, y),
                    prediction,
                )
            )

        return predictions

    def predict_grid_kriging(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        resolution: int,
        variogram_model: VariogramModel,
        max_points: int = 20,
        verbose: bool = False,
    ) -> List[
        Tuple[
            Tuple[float, float],
            Tuple[float, float],
        ]
    ]:
        predictions: List[
            Tuple[
                Tuple[float, float],
                Tuple[float, float],
            ]
        ] = []

        total = (resolution + 1) ** 2

        for index, (x, y) in enumerate(
            self.create_grid(
                xmin,
                xmax,
                ymin,
                ymax,
                resolution,
            )
        ):
            if (
                verbose
                and index
                % max(
                    1,
                    total // 10,
                )
                == 0
            ):
                print(
                    f"Progress: {index}/{total}"
                )

            value, variance = (
                self.ordinary_kriging(
                    x,
                    y,
                    variogram_model,
                    max_points=max_points,
                )
            )

            predictions.append(
                (
                    (x, y),
                    (value, variance),
                )
            )

        return predictions

    def predict_grid_block(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        resolution: int,
        block_size: float = 10.0,
    ) -> List[
        Tuple[
            Tuple[float, float],
            float,
        ]
    ]:
        if not math.isfinite(block_size) or block_size <= 0:
            raise ValueError(
                "Block size must be positive and finite"
            )

        grid_points = list(
            self.create_grid(
                xmin,
                xmax,
                ymin,
                ymax,
                resolution,
            )
        )

        block_averages = self.block_average(
            xmin,
            xmax,
            ymin,
            ymax,
            block_size,
        )

        statistics = self.statistics_summary()

        mean_value = statistics.get(
            "mean",
            0.0,
        )

        n_blocks_x, n_blocks_y = (
            self._block_shape(
                xmin,
                xmax,
                ymin,
                ymax,
                block_size,
            )
        )

        predictions: List[
            Tuple[
                Tuple[float, float],
                float,
            ]
        ] = []

        for x, y in grid_points:
            block = self._block_index(
                x,
                y,
                xmin,
                ymin,
                block_size,
                n_blocks_x,
                n_blocks_y,
            )

            prediction = block_averages.get(
                block,
                mean_value,
            )

            predictions.append(
                (
                    (x, y),
                    prediction,
                )
            )

        return predictions


if __name__ == "__main__":
    geo = SimpleGeostat(random_seed=42)

    for _ in range(200):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)

        value = (
            math.sin(x / 20)
            * math.cos(y / 20)
            + random.gauss(0, 0.1)
        )

        geo.data.append(
            DataPoint(
                x=x,
                y=y,
                value=value,
            )
        )

    geo._build_spatial_index()

    stats = geo.statistics_summary()

    print(
        f"Data points: "
        f"{stats['n_points']:.0f}"
    )

    print(
        f"Mean value: "
        f"{stats['mean']:.4f}"
    )

    print(
        f"Standard deviation: "
        f"{stats['std_dev']:.4f}"
    )

    lags, gamma = geo.experimental_variogram(
        max_lag=50,
        n_bins=15,
    )

    if len(lags) >= 3 and len(gamma) >= 3:
        spherical_model = (
            geo.fit_variogram_model(
                lags,
                gamma,
                model_type="spherical",
            )
        )

        exponential_model = (
            geo.fit_variogram_model(
                lags,
                gamma,
                model_type="exponential",
            )
        )

        gaussian_model = (
            geo.fit_variogram_model(
                lags,
                gamma,
                model_type="gaussian",
            )
        )

        print(
            "Spherical model - "
            f"Nugget: "
            f"{spherical_model.nugget:.4f}, "
            f"Partial sill: "
            f"{spherical_model.sill:.4f}, "
            f"Range: "
            f"{spherical_model.range_param:.4f}"
        )

        print(
            "Exponential model - "
            f"Nugget: "
            f"{exponential_model.nugget:.4f}, "
            f"Partial sill: "
            f"{exponential_model.sill:.4f}, "
            f"Range: "
            f"{exponential_model.range_param:.4f}"
        )

        print(
            "Gaussian model - "
            f"Nugget: "
            f"{gaussian_model.nugget:.4f}, "
            f"Partial sill: "
            f"{gaussian_model.sill:.4f}, "
            f"Range: "
            f"{gaussian_model.range_param:.4f}"
        )

        idw_rmse = geo.cross_validate_idw(
            power=2,
            k_folds=5,
            max_points=20,
        )

        kriging_rmse_fixed = (
            geo.cross_validate_kriging(
                spherical_model,
                k_folds=5,
                max_points=20,
                refit_variogram=False,
            )
        )

        kriging_rmse_refit = (
            geo.cross_validate_kriging(
                spherical_model,
                k_folds=5,
                max_points=20,
                refit_variogram=True,
            )
        )

        print(
            "IDW Cross-validation RMSE: "
            f"{idw_rmse:.4f}"
        )

        print(
            "Kriging fixed model RMSE: "
            f"{kriging_rmse_fixed:.4f}"
        )

        print(
            "Kriging refit model RMSE: "
            f"{kriging_rmse_refit:.4f}"
        )

        grid_predictions_idw = (
            geo.predict_grid_idw(
                0,
                100,
                0,
                100,
                20,
                power=2,
                max_points=10,
            )
        )

        geo.save_csv(
            "predictions_idw.csv",
            grid_predictions_idw,
        )

        grid_predictions_kriging = (
            geo.predict_grid_kriging(
                0,
                100,
                0,
                100,
                20,
                spherical_model,
                max_points=20,
            )
        )

        kriging_results = [
            (
                (x, y),
                value,
            )
            for (x, y), (value, _) in (
                grid_predictions_kriging
            )
        ]

        geo.save_csv(
            "predictions_kriging.csv",
            kriging_results,
        )

        block_predictions = (
            geo.predict_grid_block(
                0,
                100,
                0,
                100,
                10,
                block_size=10,
            )
        )

        geo.save_csv(
            "block_predictions.csv",
            block_predictions,
        )

        print(
            "IDW grid predictions saved: "
            f"{len(grid_predictions_idw)} points"
        )

        print(
            "Kriging grid predictions saved: "
            f"{len(grid_predictions_kriging)} points"
        )

        print(
            "Block predictions saved: "
            f"{len(block_predictions)} points"
        )

    else:
        print(
            "Insufficient data for variogram analysis"
        )
