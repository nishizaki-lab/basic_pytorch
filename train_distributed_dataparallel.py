import gc
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

seed = 1111
def setup_all_seed(seed=0):
    # numpyに関係する乱数シードの設定
    np.random.seed(seed)
    
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

# PyTorchで高速化するのに、必要な設定
def enable_misc_optimizations():
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark_limit = 1
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    # torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if torch.backends.cudnn.benchmark:
        print("Enabled CUDNN Benchmark Sucessfully")
    else:
        print("CUDNN Benchmark Disabled")
    if torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32:
        print("Enabled CUDA & CUDNN TF32 Sucessfully")
    else:
        print("CUDA & CUDNN TF32 Disabled")
        
enable_misc_optimizations()
setup_all_seed(seed)

# MLPモデルの定義
# シンプルな多層パーセプトロンモデルを定義するクラス
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# データローディングの設定
# 分散学習用のデータローダーを設定する関数
def get_data_loaders(batch_size, rank, world_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # 分散学習用のサンプラーを作成（各GPUに異なるデータを割り当てる）
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, 
                              num_workers=4, pin_memory=True, persistent_workers=True)
    
    # テストデータはランク0のみで評価するため、サンプラーなしで作成
    if rank == 0:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True, persistent_workers=True)
    else:
        test_loader = None
    
    return train_loader, test_loader

# 学習関数
# 分散環境でモデルを訓練するための関数
def train(model, train_loader, optimizer, criterion, device, num_epochs, rank):
    model.train()
    for epoch in range(num_epochs):
        # エポックごとにサンプラーのシードを設定（各GPUで異なるデータを使用するため）
        train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.view(inputs.size(0), -1).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # ランク0のプロセス（マスターノード）のみが進捗を表示
            if i % 100 == 99 and rank == 0:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
        
        # 各エポックの終わりにキャッシュをクリア
        torch.cuda.empty_cache()
        gc.collect()
    
    if rank == 0:
        print("学習完了")
    return model

# テスト関数を追加
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.view(inputs.size(0), -1).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'テストデータの精度: {accuracy:.2f}%')
    return accuracy

# 分散処理の初期化関数
# 各プロセスの分散環境を設定する関数
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # NCCLバックエンドを使用して分散処理グループを初期化
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 分散処理の終了関数
# 分散処理グループを破棄するクリーンアップ関数
def cleanup():
    dist.destroy_process_group()


# メイン関数
# 分散学習の設定と実行を行うメイン関数
def main(rank, world_size):
    setup(rank, world_size)
    
    # ハイパーパラメータ
    input_size = 784  # 28x28を平坦化
    hidden1_size = 128
    hidden2_size = 64
    output_size = 10
    batch_size = 16  # GPU毎のバッチサイズ
    learning_rate = 0.001
    num_epochs = 5
    model_save_path = 'mnist_model_ddp.pth'

    # デバイスの設定とcuDNNベンチマーキングの有効化
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # モデル、オプティマイザ、損失関数の作成
    model = SimpleMLP(input_size, hidden1_size, hidden2_size, output_size).to(device)
    # DistributedDataParallelでモデルをラップして分散学習に対応
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # データローダーの取得
    train_loader, test_loader = get_data_loaders(batch_size, rank, world_size)

    # 学習時間の計測
    start_time = time.time()

    # モデルの学習
    model = train(model, train_loader, optimizer, criterion, device, num_epochs, rank)

    # 学習時間の計測
    end_time = time.time()
    if rank == 0:
        print(f"学習時間: {end_time - start_time:.2f} 秒")
        
        # テストデータでの評価（ランク0のみ）
        accuracy = test(model, test_loader, device)
        
        # モデルの保存（ランク0のみ）
        # DDPでラップされているモデルからモジュールを取得
        model_to_save = model.module
        
        # モデル全体を保存
        torch.save(model_to_save.state_dict(), model_save_path)
        print(f"モデルを {model_save_path} に保存しました")
        
        # モデルの構造情報も保存
        model_info = {
            'input_size': input_size,
            'hidden1_size': hidden1_size,
            'hidden2_size': hidden2_size,
            'output_size': output_size,
            'accuracy': accuracy
        }
        torch.save(model_info, 'model_info_ddp.pth')
        print(f"モデル情報を model_info_ddp.pth に保存しました")

    cleanup()

if __name__ == "__main__":
    # 利用可能なGPUの数を取得して、その数だけプロセスを起動
    world_size = torch.cuda.device_count()
    # 複数のプロセスを起動して分散学習を実行
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)