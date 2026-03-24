import csv
import math
import random
from collections import defaultdict

class SimpleGeostat:
    
    def __init__(self):
        self.data = []
    
    def load_csv(self, filename, x_col=0, y_col=1, val_col=2, has_header=True, reset=False):
        if reset:
            self.data = []
        
        with open(filename, 'r') as f:
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
                    self.data.append((x, y, val))
                except (ValueError, IndexError):
                    continue
    
    def save_csv(self, filename, predictions):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'predicted'])
            for (x, y), pred in predictions:
                writer.writerow([x, y, pred])
    
    def create_grid(self, xmin, xmax, ymin, ymax, resolution):
        if resolution <= 0:
            raise ValueError("Resolution must be positive")
        
        x_step = (xmax - xmin) / resolution
        y_step = (ymax - ymin) / resolution
        
        grid = []
        for i in range(resolution + 1):
            for j in range(resolution + 1):
                x = xmin + i * x_step
                y = ymin + j * y_step
                grid.append((x, y))
        
        return grid
    
    def idw(self, target_x, target_y, power=2, max_points=None, data_source=None):
        if data_source is None:
            data_source = self.data
        
        distances = []
        for x, y, val in data_source:
            dist = math.hypot(x - target_x, y - target_y)
            if dist > 0:
                distances.append((dist, val))
        
        if max_points:
            distances.sort(key=lambda x: x[0])
            distances = distances[:max_points]
        
        if not distances:
            if data_source:
                return sum(v for _, _, v in data_source) / len(data_source)
            return 0
        
        total_weight = 0
        weighted_sum = 0
        for dist, val in distances:
            weight = 1 / (dist ** power)
            total_weight += weight
            weighted_sum += weight * val
        
        return weighted_sum / total_weight
    
    def block_average(self, xmin, xmax, ymin, ymax, block_size=10):
        if block_size <= 0:
            raise ValueError("Block size must be positive")
        
        blocks = defaultdict(list)
        
        for x, y, val in self.data:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                block_x = int((x - xmin) / block_size)
                block_y = int((y - ymin) / block_size)
                blocks[(block_x, block_y)].append(val)
        
        block_averages = {}
        for (bx, by), values in blocks.items():
            center_x = xmin + (bx + 0.5) * block_size
            center_y = ymin + (by + 0.5) * block_size
            block_averages[(center_x, center_y)] = sum(values) / len(values)
        
        return block_averages
    
    def experimental_variogram(self, max_lag, n_bins=20):
        if max_lag <= 0 or n_bins <= 0:
            return [], []
        
        bin_width = max_lag / n_bins
        bins = [[] for _ in range(n_bins)]
        
        n = len(self.data)
        for i in range(n):
            x1, y1, v1 = self.data[i]
            for j in range(i + 1, n):
                x2, y2, v2 = self.data[j]
                dist = math.hypot(x1 - x2, y1 - y2)
                if dist <= max_lag:
                    semivar = 0.5 * (v1 - v2) ** 2
                    bin_idx = min(int(dist / bin_width), n_bins - 1)
                    bins[bin_idx].append(semivar)
        
        lag_centers = []
        gamma = []
        for i in range(n_bins):
            if bins[i]:
                lag_centers.append((i + 0.5) * bin_width)
                gamma.append(sum(bins[i]) / len(bins[i]))
        
        return lag_centers, gamma
    
    def estimate_variogram_params(self, lags, gamma, model_type='spherical'):
        if not lags or not gamma:
            return {'nugget': 0, 'sill': 1, 'range': 10, 'model': model_type}
        
        nugget = gamma[0] if gamma else 0
        sill = gamma[-1] - nugget if gamma[-1] > nugget else 1
        variogram_range = lags[-1] * 0.7
        
        for i in range(len(lags)):
            if gamma[i] >= nugget + 0.95 * sill:
                variogram_range = lags[i]
                break
        
        if variogram_range <= 0:
            variogram_range = lags[-1] if lags else 10
        
        return {'nugget': nugget, 'sill': sill, 'range': variogram_range, 'model': model_type}
    
    def approximate_kriging_variance(self, target_x, target_y, variogram_params, max_points=20):
        distances = []
        for x, y, val in self.data:
            dist = math.hypot(x - target_x, y - target_y)
            distances.append((dist, val, x, y))
        
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:max_points]
        
        if not neighbors:
            return variogram_params['sill'] + variogram_params['nugget']
        
        n_neighbors = len(neighbors)
        A = []
        for i in range(n_neighbors):
            row = []
            for j in range(n_neighbors):
                dist_ij = math.hypot(neighbors[i][2] - neighbors[j][2], neighbors[i][3] - neighbors[j][3])
                gamma_val = self._variogram_value(dist_ij, variogram_params)
                row.append(gamma_val)
            row.append(1)
            A.append(row)
        
        last_row = [1] * n_neighbors + [0]
        A.append(last_row)
        
        b = []
        for i in range(n_neighbors):
            di = neighbors[i][0]
            gamma_val = self._variogram_value(di, variogram_params)
            b.append(gamma_val)
        b.append(1)
        
        try:
            n = n_neighbors + 1
            augmented = [A[i] + [b[i]] for i in range(n)]
            
            for col in range(n):
                pivot = augmented[col][col]
                if abs(pivot) < 1e-10:
                    continue
                
                for row in range(n):
                    if row != col:
                        factor = augmented[row][col] / pivot
                        for k in range(n + 1):
                            augmented[row][k] -= factor * augmented[col][k]
                
                for k in range(n + 1):
                    augmented[col][k] /= pivot
            
            weights = [augmented[i][n] for i in range(n_neighbors)]
            lagrange_multiplier = augmented[n_neighbors][n] if n_neighbors < n else 0
            
            kriging_variance = lagrange_multiplier
            
            for i in range(n_neighbors):
                for j in range(n_neighbors):
                    dist_ij = math.hypot(neighbors[i][2] - neighbors[j][2], neighbors[i][3] - neighbors[j][3])
                    gamma_ij = self._variogram_value(dist_ij, variogram_params)
                    kriging_variance -= weights[i] * weights[j] * gamma_ij
            
            return max(0, kriging_variance)
        except Exception:
            return variogram_params['sill'] + variogram_params['nugget']
    
    def _variogram_value(self, h, params):
        nugget = params['nugget']
        sill = params['sill']
        r = params['range']
        model = params.get('model', 'spherical')
        
        if h <= 0:
            return 0
        
        if r <= 0:
            return nugget + sill
        
        if model == 'spherical':
            if h < r:
                return nugget + sill * (1.5 * h / r - 0.5 * (h / r) ** 3)
            else:
                return nugget + sill
        elif model == 'exponential':
            return nugget + sill * (1 - math.exp(-3 * h / r))
        else:
            if h < r:
                return nugget + sill * (h / r)
            else:
                return nugget + sill
    
    def cross_validate(self, power=2, k_folds=5):
        if not self.data or k_folds <= 0:
            return 0
        
        k_folds = min(k_folds, len(self.data))
        shuffled_data = self.data[:]
        random.shuffle(shuffled_data)
        fold_size = max(1, len(shuffled_data) // k_folds)
        errors = []
        
        for fold in range(k_folds):
            start = fold * fold_size
            end = start + fold_size if fold < k_folds - 1 else len(shuffled_data)
            
            test_set = shuffled_data[start:end]
            train_set = shuffled_data[:start] + shuffled_data[end:]
            
            for x, y, true_val in test_set:
                predicted = self.idw(x, y, power, data_source=train_set)
                error = (true_val - predicted) ** 2
                errors.append(error)
        
        rmse = math.sqrt(sum(errors) / len(errors)) if errors else 0
        return rmse
    
    def statistics_summary(self):
        if not self.data:
            return {}
        
        values = [v for _, _, v in self.data]
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_dev = math.sqrt(variance)
        
        x_coords = [x for x, _, _ in self.data]
        y_coords = [y for _, y, _ in self.data]
        
        return {
            'n_points': len(self.data),
            'mean': mean_val,
            'variance': variance,
            'std_dev': std_dev,
            'min': min(values),
            'max': max(values),
            'x_range': (min(x_coords), max(x_coords)),
            'y_range': (min(y_coords), max(y_coords))
        }
    
    def predict_grid_idw(self, xmin, xmax, ymin, ymax, resolution, power=2, max_points=None):
        grid = self.create_grid(xmin, xmax, ymin, ymax, resolution)
        predictions = []
        
        for x, y in grid:
            pred = self.idw(x, y, power=power, max_points=max_points)
            predictions.append(((x, y), pred))
        
        return predictions
    
    def predict_grid_block(self, xmin, xmax, ymin, ymax, resolution, block_size=10):
        grid = self.create_grid(xmin, xmax, ymin, ymax, resolution)
        block_averages = self.block_average(xmin, xmax, ymin, ymax, block_size)
        mean_value = self.statistics_summary().get('mean', 0)
        predictions = []
        
        for x, y in grid:
            block_x = int((x - xmin) / block_size)
            block_y = int((y - ymin) / block_size)
            center_x = block_x * block_size + xmin + block_size / 2
            center_y = block_y * block_size + ymin + block_size / 2
            pred = block_averages.get((center_x, center_y), mean_value)
            predictions.append(((x, y), pred))
        
        return predictions

if __name__ == "__main__":
    geo = SimpleGeostat()
    
    for i in range(200):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        val = math.sin(x/20) * math.cos(y/20) + random.gauss(0, 0.1)
        geo.data.append((x, y, val))
    
    stats = geo.statistics_summary()
    print(f"Data points: {stats['n_points']}")
    print(f"Mean value: {stats['mean']:.4f}")
    print(f"Standard deviation: {stats['std_dev']:.4f}")
    
    lags, gamma = geo.experimental_variogram(max_lag=50, n_bins=15)
    variogram_params = geo.estimate_variogram_params(lags, gamma)
    print(f"Variogram nugget: {variogram_params['nugget']:.4f}")
    print(f"Variogram sill: {variogram_params['sill']:.4f}")
    print(f"Variogram range: {variogram_params['range']:.4f}")
    
    rmse = geo.cross_validate(power=2, k_folds=5)
    print(f"Cross-validation RMSE: {rmse:.4f}")
    
    grid_predictions = geo.predict_grid_idw(0, 100, 0, 100, 20, power=2, max_points=10)
    geo.save_csv('predictions_full.csv', grid_predictions)
    
    block_predictions = geo.predict_grid_block(0, 100, 0, 100, 10, block_size=10)
    geo.save_csv('block_predictions.csv', block_predictions)
    
    print(f"Grid predictions saved: {len(grid_predictions)} points")
    print(f"Block predictions saved: {len(block_predictions)} points")
