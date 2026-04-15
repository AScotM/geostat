import csv
import math
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Generator
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from dataclasses import dataclass
from copy import deepcopy

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
    
    def load_csv(self, filename: str, x_col: int = 0, y_col: int = 1, 
                 val_col: int = 2, has_header: bool = True, reset: bool = False) -> None:
        if reset:
            self.clear_data()
        
        with open(filename, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            if has_header:
                try:
                    next(reader)
                except StopIteration:
                    pass
            
            for row in reader:
                try:
                    x = float(row[x_col])
                    y = float(row[y_col])
                    val = float(row[val_col])
                    self.data.append(DataPoint(x=x, y=y, value=val))
                except (ValueError, IndexError):
                    continue
        
        self._build_spatial_index()
    
    def _build_spatial_index(self) -> None:
        if self.data:
            points = np.array([[p.x, p.y] for p in self.data])
            self.kdtree = cKDTree(points)
            self.data_array = np.array([[p.x, p.y, p.value] for p in self.data])
    
    def save_csv(self, filename: str, predictions: List[Tuple[Tuple[float, float], float]]) -> None:
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'predicted'])
            for (x, y), pred in predictions:
                writer.writerow([x, y, pred])
    
    def clear_data(self) -> None:
        self.data = []
        self.kdtree = None
        self.data_array = None
    
    def create_grid(self, xmin: float, xmax: float, ymin: float, ymax: float, 
                    resolution: int) -> Generator[Tuple[float, float], None, None]:
        if xmin >= xmax:
            raise ValueError("xmin must be less than xmax")
        if ymin >= ymax:
            raise ValueError("ymin must be less than ymax")
        if resolution <= 0:
            raise ValueError("Resolution must be positive")
        
        x_step = (xmax - xmin) / resolution
        y_step = (ymax - ymin) / resolution
        
        eps = 1e-10
        for i in range(resolution + 1):
            for j in range(resolution + 1):
                x = xmin + i * x_step
                y = ymin + j * y_step
                if x > xmax + eps or y > ymax + eps:
                    continue
                if x > xmax:
                    x = xmax
                if y > ymax:
                    y = ymax
                yield (x, y)
    
    def idw(self, target_x: float, target_y: float, power: float = 2, 
            max_points: Optional[int] = None, data_source: Optional[List[DataPoint]] = None) -> float:
        if data_source is None:
            if self.kdtree is None or self.data_array is None:
                return 0.0
            if max_points:
                distances, indices = self.kdtree.query([target_x, target_y], k=max_points)
                if np.isscalar(distances):
                    distances = np.array([distances])
                    indices = np.array([indices])
                neighbors = self.data_array[indices]
            else:
                neighbors = self.data_array
                distances = np.sqrt((neighbors[:, 0] - target_x)**2 + (neighbors[:, 1] - target_y)**2)
            
            zero_dist_mask = distances == 0
            if np.any(zero_dist_mask):
                return float(neighbors[zero_dist_mask][0, 2])
            
            weights = 1.0 / (distances ** power)
            total_weight = np.sum(weights)
            if total_weight == 0:
                return float(np.mean(neighbors[:, 2]))
            
            weighted_sum = np.sum(weights * neighbors[:, 2])
            return float(weighted_sum / total_weight)
        
        else:
            distances = []
            for point in data_source:
                dist = math.hypot(point.x - target_x, point.y - target_y)
                if dist == 0:
                    return point.value
                distances.append((dist, point.value))
            
            if max_points:
                distances.sort(key=lambda item: item[0])
                distances = distances[:max_points]
            
            if not distances:
                if data_source:
                    return sum(p.value for p in data_source) / len(data_source)
                return 0
            
            total_weight = 0
            weighted_sum = 0
            for dist, val in distances:
                weight = 1 / (dist ** power)
                total_weight += weight
                weighted_sum += weight * val
            
            return weighted_sum / total_weight
    
    def block_average(self, xmin: float, xmax: float, ymin: float, ymax: float, 
                      block_size: float = 10) -> Dict[Tuple[float, float], float]:
        if block_size <= 0:
            raise ValueError("Block size must be positive")
        if xmin >= xmax:
            raise ValueError("xmin must be less than xmax")
        if ymin >= ymax:
            raise ValueError("ymin must be less than ymax")
        
        blocks = defaultdict(list)
        
        for point in self.data:
            if xmin <= point.x <= xmax and ymin <= point.y <= ymax:
                block_x = int((point.x - xmin) // block_size)
                block_y = int((point.y - ymin) // block_size)
                blocks[(block_x, block_y)].append(point.value)
        
        block_averages = {}
        eps = 1e-10
        for (bx, by), values in blocks.items():
            center_x = xmin + (bx + 0.5) * block_size
            center_y = ymin + (by + 0.5) * block_size
            if center_x <= xmax + eps and center_y <= ymax + eps:
                block_averages[(center_x, center_y)] = sum(values) / len(values)
        
        return block_averages
    
    def experimental_variogram(self, max_lag: float, n_bins: int = 20) -> Tuple[List[float], List[float]]:
        if max_lag <= 0 or n_bins <= 0 or len(self.data) < 2:
            return [], []
        
        bin_width = max_lag / n_bins
        bins = [[] for _ in range(n_bins)]
        
        n = len(self.data)
        for i in range(n):
            p1 = self.data[i]
            for j in range(i + 1, n):
                p2 = self.data[j]
                dist = math.hypot(p1.x - p2.x, p1.y - p2.y)
                if dist <= max_lag:
                    semivar = 0.5 * (p1.value - p2.value) ** 2
                    bin_idx = min(int(dist / bin_width), n_bins - 1)
                    bins[bin_idx].append(semivar)
        
        lag_centers = []
        gamma = []
        for i in range(n_bins):
            if bins[i]:
                lag_centers.append((i + 0.5) * bin_width)
                gamma.append(sum(bins[i]) / len(bins[i]))
        
        return lag_centers, gamma
    
    def _spherical_variogram(self, h: float, nugget: float, sill: float, range_param: float) -> float:
        if h <= 0:
            return 0
        if range_param <= 0:
            return nugget + sill
        if h < range_param:
            return nugget + sill * (1.5 * h / range_param - 0.5 * (h / range_param) ** 3)
        else:
            return nugget + sill
    
    def _exponential_variogram(self, h: float, nugget: float, sill: float, range_param: float) -> float:
        if h <= 0:
            return 0
        if range_param <= 0:
            return nugget + sill
        return nugget + sill * (1 - math.exp(-3 * h / range_param))
    
    def _gaussian_variogram(self, h: float, nugget: float, sill: float, range_param: float) -> float:
        if h <= 0:
            return 0
        if range_param <= 0:
            return nugget + sill
        return nugget + sill * (1 - math.exp(-3 * (h / range_param) ** 2))
    
    def fit_variogram_model(self, lags: List[float], gamma: List[float], 
                           model_type: str = 'spherical') -> VariogramModel:
        if len(lags) < 3 or len(gamma) < 3:
            return VariogramModel(nugget=0.0, sill=1.0, range_param=10.0, model_type=model_type)
        
        lags_array = np.array(lags)
        gamma_array = np.array(gamma)
        
        nugget_initial = max(0.0, gamma_array[0])
        sill_initial = max(0.01, gamma_array[-1] - nugget_initial)
        
        if sill_initial <= 1e-6:
            sill_initial = 1.0
        
        range_initial = max(0.1, lags_array[-1] * 0.5)
        
        if model_type == 'spherical':
            def model_func(h, nugget, sill, r):
                result = np.zeros_like(h, dtype=np.float64)
                r_safe = max(r, 1e-8)
                mask = h < r_safe
                result[mask] = nugget + sill * (1.5 * h[mask] / r_safe - 0.5 * (h[mask] / r_safe) ** 3)
                result[~mask] = nugget + sill
                return result
        elif model_type == 'exponential':
            def model_func(h, nugget, sill, r):
                r_safe = max(r, 1e-8)
                return nugget + sill * (1 - np.exp(-3 * h / r_safe))
        elif model_type == 'gaussian':
            def model_func(h, nugget, sill, r):
                r_safe = max(r, 1e-8)
                return nugget + sill * (1 - np.exp(-3 * (h / r_safe) ** 2))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        try:
            popt, _ = curve_fit(model_func, lags_array, gamma_array, 
                               p0=[nugget_initial, sill_initial, range_initial],
                               bounds=([0, 0, 0.01], [np.inf, np.inf, np.inf]),
                               maxfev=10000)
            nugget, sill, range_param = popt
            sill = max(sill, 0.01)
            nugget = max(0, nugget)
            range_param = max(0.1, range_param)
            return VariogramModel(nugget=nugget, sill=sill, range_param=range_param, model_type=model_type)
        except Exception:
            return VariogramModel(nugget=nugget_initial, sill=sill_initial, 
                                 range_param=range_initial, model_type=model_type)
    
    def _variogram_value(self, h: float, model: VariogramModel) -> float:
        if model.model_type == 'spherical':
            return self._spherical_variogram(h, model.nugget, model.sill, model.range_param)
        elif model.model_type == 'exponential':
            return self._exponential_variogram(h, model.nugget, model.sill, model.range_param)
        elif model.model_type == 'gaussian':
            return self._gaussian_variogram(h, model.nugget, model.sill, model.range_param)
        else:
            return model.nugget + model.sill
    
    def ordinary_kriging(self, target_x: float, target_y: float, 
                        variogram_model: VariogramModel, max_points: int = 20) -> Tuple[float, float]:
        if self.kdtree is None or self.data_array is None or len(self.data_array) == 0:
            return 0.0, variogram_model.nugget + variogram_model.sill
        
        n_total = len(self.data_array)
        k = min(max_points, n_total)
        distances, indices = self.kdtree.query([target_x, target_y], k=k)
        
        if np.isscalar(distances):
            distances = np.array([distances])
            indices = np.array([indices])
        
        if np.any(distances < 1e-12):
            zero_idx = indices[distances < 1e-12][0]
            return float(self.data_array[zero_idx, 2]), 0.0
        
        neighbors = self.data_array[indices]
        n_neighbors = len(neighbors)
        
        gamma_matrix = np.zeros((n_neighbors, n_neighbors))
        for i in range(n_neighbors):
            for j in range(n_neighbors):
                if i == j:
                    gamma_matrix[i, j] = 0
                else:
                    h = np.hypot(neighbors[i, 0] - neighbors[j, 0], neighbors[i, 1] - neighbors[j, 1])
                    gamma_matrix[i, j] = self._variogram_value(h, variogram_model)
        
        A = np.zeros((n_neighbors + 1, n_neighbors + 1))
        A[:n_neighbors, :n_neighbors] = gamma_matrix
        A[:n_neighbors, n_neighbors] = 1
        A[n_neighbors, :n_neighbors] = 1
        A[n_neighbors, n_neighbors] = 0
        
        b = np.zeros(n_neighbors + 1)
        for i in range(n_neighbors):
            h = distances[i]
            b[i] = self._variogram_value(h, variogram_model)
        b[n_neighbors] = 1
        
        try:
            weights = np.linalg.solve(A, b)
            kriging_weights = weights[:n_neighbors]
            lagrange_multiplier = weights[n_neighbors]
            estimated_value = np.sum(kriging_weights * neighbors[:, 2])
            kriging_variance = np.sum(kriging_weights * b[:n_neighbors]) + lagrange_multiplier
            return float(estimated_value), max(0.0, float(kriging_variance))
        except np.linalg.LinAlgError:
            A_reg = A.copy()
            A_reg[:n_neighbors, :n_neighbors] += np.eye(n_neighbors) * 1e-8
            try:
                weights = np.linalg.solve(A_reg, b)
                kriging_weights = weights[:n_neighbors]
                lagrange_multiplier = weights[n_neighbors]
                estimated_value = np.sum(kriging_weights * neighbors[:, 2])
                kriging_variance = np.sum(kriging_weights * b[:n_neighbors]) + lagrange_multiplier
                return float(estimated_value), max(0.0, float(kriging_variance))
            except np.linalg.LinAlgError:
                weights = np.ones(n_neighbors) / n_neighbors
                estimated_value = np.sum(weights * neighbors[:, 2])
                kriging_variance = variogram_model.nugget + variogram_model.sill
                return float(estimated_value), float(kriging_variance)
    
    def cross_validate_idw(self, power: float = 2, k_folds: int = 5) -> float:
        if len(self.data) < 2 or k_folds <= 0:
            return 0.0
        
        k_folds = min(k_folds, len(self.data))
        if self.random_seed is not None:
            rng = random.Random(self.random_seed)
            shuffled_data = self.data[:]
            rng.shuffle(shuffled_data)
        else:
            shuffled_data = self.data[:]
            random.shuffle(shuffled_data)
        
        fold_size = max(1, len(shuffled_data) // k_folds)
        errors = []
        
        for fold in range(k_folds):
            start = fold * fold_size
            end = start + fold_size if fold < k_folds - 1 else len(shuffled_data)
            
            test_set = shuffled_data[start:end]
            train_set = shuffled_data[:start] + shuffled_data[end:]
            
            for point in test_set:
                predicted = self.idw(point.x, point.y, power, data_source=train_set)
                error = (point.value - predicted) ** 2
                errors.append(error)
        
        rmse = math.sqrt(sum(errors) / len(errors)) if errors else 0
        return rmse
    
    def cross_validate_kriging(self, variogram_model: VariogramModel, 
                               k_folds: int = 5, max_points: int = 20) -> float:
        if len(self.data) < 2 or k_folds <= 0:
            return 0.0
        
        k_folds = min(k_folds, len(self.data))
        if self.random_seed is not None:
            rng = random.Random(self.random_seed)
            shuffled_data = self.data[:]
            rng.shuffle(shuffled_data)
        else:
            shuffled_data = self.data[:]
            random.shuffle(shuffled_data)
        
        fold_size = max(1, len(shuffled_data) // k_folds)
        errors = []
        
        for fold in range(k_folds):
            start = fold * fold_size
            end = start + fold_size if fold < k_folds - 1 else len(shuffled_data)
            
            test_set = shuffled_data[start:end]
            train_set = shuffled_data[:start] + shuffled_data[end:]
            
            temp_geo = SimpleGeostat(random_seed=self.random_seed)
            temp_geo.data = deepcopy(train_set)
            temp_geo._build_spatial_index()
            
            for point in test_set:
                predicted, _ = temp_geo.ordinary_kriging(point.x, point.y, variogram_model, max_points)
                error = (point.value - predicted) ** 2
                errors.append(error)
        
        rmse = math.sqrt(sum(errors) / len(errors)) if errors else 0
        return rmse
    
    def statistics_summary(self) -> Dict[str, float]:
        if not self.data:
            return {}
        
        values = [p.value for p in self.data]
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_dev = math.sqrt(variance)
        
        x_coords = [p.x for p in self.data]
        y_coords = [p.y for p in self.data]
        
        return {
            'n_points': float(len(self.data)),
            'mean': mean_val,
            'variance': variance,
            'std_dev': std_dev,
            'min': min(values),
            'max': max(values),
            'x_range_min': min(x_coords),
            'x_range_max': max(x_coords),
            'y_range_min': min(y_coords),
            'y_range_max': max(y_coords)
        }
    
    def predict_grid_idw(self, xmin: float, xmax: float, ymin: float, ymax: float, 
                         resolution: int, power: float = 2, max_points: Optional[int] = None,
                         verbose: bool = False) -> List[Tuple[Tuple[float, float], float]]:
        grid_generator = self.create_grid(xmin, xmax, ymin, ymax, resolution)
        predictions = []
        
        total = (resolution + 1) ** 2
        for idx, (x, y) in enumerate(grid_generator):
            if verbose and idx % max(1, total // 10) == 0:
                print(f"Progress: {idx}/{total}")
            pred = self.idw(x, y, power=power, max_points=max_points)
            predictions.append(((x, y), pred))
        
        return predictions
    
    def predict_grid_kriging(self, xmin: float, xmax: float, ymin: float, ymax: float,
                            resolution: int, variogram_model: VariogramModel, 
                            max_points: int = 20, verbose: bool = False) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        grid_generator = self.create_grid(xmin, xmax, ymin, ymax, resolution)
        predictions = []
        
        total = (resolution + 1) ** 2
        for idx, (x, y) in enumerate(grid_generator):
            if verbose and idx % max(1, total // 10) == 0:
                print(f"Progress: {idx}/{total}")
            pred_value, pred_variance = self.ordinary_kriging(x, y, variogram_model, max_points)
            predictions.append(((x, y), (pred_value, pred_variance)))
        
        return predictions
    
    def predict_grid_block(self, xmin: float, xmax: float, ymin: float, ymax: float, 
                          resolution: int, block_size: float = 10) -> List[Tuple[Tuple[float, float], float]]:
        grid_points = list(self.create_grid(xmin, xmax, ymin, ymax, resolution))
        block_averages = self.block_average(xmin, xmax, ymin, ymax, block_size)
        mean_value = self.statistics_summary().get('mean', 0)
        predictions = []
        
        for x, y in grid_points:
            if x < xmin or x > xmax or y < ymin or y > ymax:
                continue
            block_x = int((x - xmin) // block_size)
            block_y = int((y - ymin) // block_size)
            center_x = xmin + (block_x + 0.5) * block_size
            center_y = ymin + (block_y + 0.5) * block_size
            if center_x <= xmax + 1e-10 and center_y <= ymax + 1e-10:
                pred = block_averages.get((center_x, center_y), mean_value)
            else:
                pred = mean_value
            predictions.append(((x, y), pred))
        
        return predictions


if __name__ == "__main__":
    geo = SimpleGeostat(random_seed=42)
    
    for i in range(200):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        val = math.sin(x/20) * math.cos(y/20) + random.gauss(0, 0.1)
        geo.data.append(DataPoint(x=x, y=y, value=val))
    
    geo._build_spatial_index()
    
    stats = geo.statistics_summary()
    print(f"Data points: {stats['n_points']:.0f}")
    print(f"Mean value: {stats['mean']:.4f}")
    print(f"Standard deviation: {stats['std_dev']:.4f}")
    
    lags, gamma = geo.experimental_variogram(max_lag=50, n_bins=15)
    if lags and gamma:
        spherical_model = geo.fit_variogram_model(lags, gamma, model_type='spherical')
        exponential_model = geo.fit_variogram_model(lags, gamma, model_type='exponential')
        
        print(f"Spherical model - Nugget: {spherical_model.nugget:.4f}, Sill: {spherical_model.sill:.4f}, Range: {spherical_model.range_param:.4f}")
        print(f"Exponential model - Nugget: {exponential_model.nugget:.4f}, Sill: {exponential_model.sill:.4f}, Range: {exponential_model.range_param:.4f}")
        
        idw_rmse = geo.cross_validate_idw(power=2, k_folds=5)
        kriging_rmse = geo.cross_validate_kriging(spherical_model, k_folds=5, max_points=20)
        
        print(f"IDW Cross-validation RMSE: {idw_rmse:.4f}")
        print(f"Kriging Cross-validation RMSE: {kriging_rmse:.4f}")
        
        grid_predictions_idw = geo.predict_grid_idw(0, 100, 0, 100, 20, power=2, max_points=10)
        geo.save_csv('predictions_idw.csv', grid_predictions_idw)
        
        grid_predictions_kriging = geo.predict_grid_kriging(0, 100, 0, 100, 20, spherical_model, max_points=20)
        kriging_results = [((x, y), value) for (x, y), (value, _) in grid_predictions_kriging]
        geo.save_csv('predictions_kriging.csv', kriging_results)
        
        block_predictions = geo.predict_grid_block(0, 100, 0, 100, 10, block_size=10)
        geo.save_csv('block_predictions.csv', block_predictions)
        
        print(f"IDW grid predictions saved: {len(grid_predictions_idw)} points")
        print(f"Kriging grid predictions saved: {len(grid_predictions_kriging)} points")
        print(f"Block predictions saved: {len(block_predictions)} points")
    else:
        print("Insufficient data for variogram analysis")
