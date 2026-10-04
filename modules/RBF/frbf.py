import torch
import torch.nn.functional as F
import gc, time
import subprocess


def get_gpu_temp():
    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', 
                           '--format=csv,noheader'], 
                           capture_output=True, text=True)
    return int(result.stdout.strip())

def construct_phi(  image, sigma, NodeX1, NodeX2, batch_size=None,
                    pixel_pos=None, center_pos=None, threshold=1e-9):

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X1, X2 = image.shape
    if center_pos is None:
        step_x = max(1, X1 // NodeX1)
        step_y = max(1, X2 // NodeX2)
        
        x_coords = torch.arange(0, X1, step_x, dtype=torch.float32, device=device)
        y_coords = torch.arange(0, X2, step_y, dtype=torch.float32, device=device)
        
        xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
        node_centers = torch.stack([xx.ravel(), yy.ravel()], dim=-1)
        
        del xx, yy, x_coords, y_coords
    else:
        node_centers = torch.as_tensor(center_pos, dtype=torch.float32, device=device)
    
    nIn = node_centers.shape[0]
    targetSp_tensor = torch.as_tensor(image, dtype=torch.float32, device=device)
    
    if batch_size is None:
        batch_size = nIn
    
    row_indices = torch.arange(X1, dtype=torch.float32, device=device)  # [X1]
    col_indices = torch.arange(X2, dtype=torch.float32, device=device)  # [X2]
    
    batched_netSp_sparse = []
    
    for i in range(0, nIn, batch_size):
        this_batch_size = min(batch_size, nIn - i)
        batch_centers = node_centers[i:i+this_batch_size]
        
        #sparse_matrices = []

        cx = batch_centers[:, 0].unsqueeze(1)  # [batch_size, 1]
        cy = batch_centers[:, 1].unsqueeze(1)  # [batch_size, 1]

        row_diff = row_indices.unsqueeze(0).unsqueeze(2) - cx.unsqueeze(1)  # [batch_size, X1, 1]
        col_diff = col_indices.unsqueeze(0).unsqueeze(1) - cy.unsqueeze(1)  # [batch_size, 1, X2]

        dist_sq = row_diff**2 + col_diff**2 
        
        rbf_values = torch.exp(-dist_sq / (2 * sigma * sigma))
        if not threshold is None:
            rbf_values[rbf_values < threshold] = 0

        sparse_batch = rbf_values.to_sparse()
        batched_netSp_sparse.append(sparse_batch)
        
        del batch_centers, sparse_batch
    
    del node_centers, row_indices, col_indices
    torch.cuda.empty_cache()
    
    return batched_netSp_sparse, targetSp_tensor

class WeightCalculator:
    def __init__(self, nIn, input_w=None, device='cuda'):
        self.nIn = nIn
        self.device = device
        
        if input_w is None:
            #self.weights = (torch.rand(nIn, dtype=torch.float32, device=device) * 0.1 - 0.05)
            self.weights = torch.zeros(nIn, dtype=torch.float32, device=device) + 1e-6
            #self.weights = torch.rand(nIn, dtype=torch.float32, device=device)
        else:
            self.weights = torch.as_tensor(input_w, dtype=torch.float32).to(device)
    
    def update_batch(self, netSp_dense, targetSp, teta):
        # Переводим в [X1*X2, batch_size] для matmul
        pixel_nod_grid = netSp_dense.flatten(1, 2).transpose(0, 1)  # [X1*X2, batch_size]
        
        y_pred = torch.matmul(pixel_nod_grid, self.weights)  # [X1*X2]
        
        delta = targetSp - y_pred  # [X1*X2]
        
        # Градиент: [batch_size] = [batch_size, X1*X2] @ [X1*X2] / nIn
        gradient = torch.matmul(pixel_nod_grid.transpose(0, 1), delta) / self.nIn

        self.weights += teta * gradient

        return torch.norm(gradient)
    
    def get_weights(self):
        return self.weights


def train_weights(netSp_sparse, targetSp_tensor, nIterations, start_teta, 
                           decay_rate=0.99, input_w=None, min_teta=0.01, debug=False):
    """
    Полностью векторизованное обучение весов
    
    Args:
        netSp_sparse: список sparse тензоров [batch_i, X1, X2]
        targetSp_tensor: [X1*X2] - целевые значения (оригинальное изображение)
        nIterations: количество итераций
        start_teta: начальный learning rate
        decay_rate: коэффициент затухания
        input_w: начальные веса (опционально)
        min_teta: минимальное learning rate
    
    Returns:
        seq_weights: список весов для всех узлов
    """
    gc.disable()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not isinstance(targetSp_tensor, torch.Tensor):
        targetSp_tensor = torch.as_tensor(targetSp_tensor, dtype=torch.float32, device=device)
    target_flat = targetSp_tensor.flatten()  # [X1*X2]
    
    # Предварительно вычисляем learning rates для всех итераций
    tetas = torch.tensor([start_teta * (decay_rate ** i) for i in range(nIterations)], 
                         dtype=torch.float32, device=device)
    tetas = torch.clamp(tetas, min=min_teta)
    
    seq_weights = []
    count = 0
    timing_info = []
    overall_start = time.time()

    torch.cuda.empty_cache()  # Очищаем кэш перед началом
    torch.cuda.synchronize()

    for batch_idx, batch_sparse in enumerate(netSp_sparse):
        batch_start = time.time()
        batch_size = batch_sparse.shape[0]

        if debug:
            mem_before = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
        # можно давать начальные значения для ускорения
        # но на данный момент неясно, есть ли в этом смысл
        if input_w is not None and count < len(input_w):
            end_idx = min(count + batch_size, len(input_w))
            batch_input_w = input_w[count:end_idx]
            if len(batch_input_w) < batch_size:
                padding = torch.rand(batch_size - len(batch_input_w), device=device) * 0.1 - 0.05
                batch_input_w = torch.cat([batch_input_w, padding])
        else:
            batch_input_w = None
        
        # Создаем модель для батча
        model = WeightCalculator(batch_size, input_w=batch_input_w, device=device)
        
        convert_start = time.time()
        batch_dense = batch_sparse.to_dense() # [batch_size, X1, X2]
        #torch.cuda.synchronize()
        convert_time = time.time() - convert_start
        
        # Векторизованное обучение - цикл только по итерациям (неизбежно из-за изменения LR)
        train_start = time.time()
        min_loss = torch.as_tensor(1000, dtype=torch.float32, device=device)
        optimal_weights = None
        for i in range(nIterations):
            loss = model.update_batch(batch_dense, target_flat, tetas[i])
            if loss < min_loss:
                min_loss = loss
                optimal_weights = model.get_weights()
        #torch.cuda.synchronize()
        train_time = time.time() - train_start
        
        # Сохраняем веса
        weights = optimal_weights if optimal_weights is not None else model.get_weights()
        seq_weights.append(weights)
        batch_time = time.time() - batch_start
        count += batch_size

        if debug:
            mem_after = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            timing_info.append({
                'batch_idx': batch_idx,
                'batch_size': batch_size,
                'convert_time': convert_time,
                'train_time': train_time,
                'total_time': batch_time,
                'mem_before_gb': mem_before,
                'mem_after_gb': mem_after,
                'mem_diff_gb': mem_after - mem_before,
                'min_loss': min_loss,
                'final_loss': loss
            })
            
            print(f"Batch {batch_idx}: {batch_time:.2f}s "
                  f"(convert: {convert_time:.2f}s, train: {train_time:.2f}s) "
                  f"Mem: {mem_before:.2f}GB → {mem_after:.2f}GB")
            
            if batch_idx % 10 == 0:  # Не каждый раз, чтобы не замедлять
                        print(f"Температура GPU: {get_gpu_temp()}C")
                    #    torch.cuda.empty_cache()
        
        del model, batch_dense
        
    
    overall_time = time.time() - overall_start
    
    # Включаем GC обратно
    gc.enable()
    gc.collect()
    
    all_weights = torch.cat(seq_weights).cpu().numpy().tolist()
    
    if debug:
        print(f"\nОбщее время: {overall_time:.2f}s")
        print(f"Среднее время на батч: {overall_time/len(netSp_sparse):.2f}s")
        print(f"Медленные батчи (>2x среднего):")
        avg_time = overall_time / len(netSp_sparse)
        for info in timing_info:
            if info['total_time'] > 2 * avg_time:
                print(f"  Batch {info['batch_idx']}: {info['total_time']:.2f}s "
                      f"(mem diff: {info['mem_diff_gb']:+.2f}GB)")
    torch.cuda.empty_cache()
    return all_weights