import gc
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
    # TF32を有効化して計算を高速化（精度をわずかに犠牲にする）
    torch.backends.cudnn.allow_tf32 = True
    # ベンチマークを無効化して初期化時間を短縮
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark_limit = 1
    # 行列演算でTF32を有効化
    torch.backends.cuda.matmul.allow_tf32 = True
    # FP16の精度低減を無効化
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
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        # シンプルな3層ニューラルネットワークの初期化
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        # 順伝播の定義：入力を受け取り、各層を通して出力を計算
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# データローディングの設定
def get_data_loaders(batch_size):
    # MNISTデータセットの前処理とデータローダーの設定
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    # 訓練データとテストデータのダウンロードと変換
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # データローダーの作成（マルチスレッド処理とGPUへの効率的な転送を設定）
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    
    return train_loader, test_loader

# 学習関数
def train(model, train_loader, optimizer, criterion, device, num_epochs):
    # モデルを訓練モードに設定
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # データをGPUに転送（non_blockingで非同期転送）
            inputs = inputs.view(inputs.size(0), -1).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 勾配をゼロに初期化（メモリ効率のためにset_to_none=True）
            optimizer.zero_grad(set_to_none=True)
            # 順伝播
            outputs = model(inputs)
            # 損失計算
            loss = criterion(outputs, labels)
            
            # 逆伝播で勾配計算
            loss.backward()
            # パラメータ更新
            optimizer.step()
            
            # 損失の累積と表示
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
        
        # 各エポックの終わりにキャッシュをクリア
        torch.cuda.empty_cache()
        gc.collect()
    
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

# 学習の設定と実行のためのメイン関数
def main():
    # ハイパーパラメータの設定
    input_size = 784  # 28x28を平坦化
    hidden1_size = 128
    hidden2_size = 64
    output_size = 10
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 5
    model_save_path = 'mnist_model.pth'

    # デバイスの設定とcuDNNベンチマーキングの有効化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    torch.backends.cudnn.benchmark = True

    # モデル、オプティマイザ、損失関数の作成
    model = SimpleMLP(input_size, hidden1_size, hidden2_size, output_size).to(device)
    
    # Adamオプティマイザの設定
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 交差エントロピー損失関数の設定
    criterion = nn.CrossEntropyLoss()

    # データローダーの取得
    train_loader, test_loader = get_data_loaders(batch_size)

    # 学習時間の計測開始
    start_time = time.time()

    # モデルの学習実行
    model = train(model, train_loader, optimizer, criterion, device, num_epochs)

    # 学習時間の計測終了と表示
    end_time = time.time()
    print(f"学習時間: {end_time - start_time:.2f} 秒")
    
    # テストデータでの評価
    accuracy = test(model, test_loader, device)
    
    # モデルの保存
    # モデル全体を保存
    torch.save(model.state_dict(), model_save_path)
    print(f"モデルを {model_save_path} に保存しました")
    
    # モデルの構造情報も保存
    model_info = {
        'input_size': input_size,
        'hidden1_size': hidden1_size,
        'hidden2_size': hidden2_size,
        'output_size': output_size,
        'accuracy': accuracy
    }
    torch.save(model_info, 'model_info.pth')
    print(f"モデル情報を model_info.pth に保存しました")

if __name__ == "__main__":
    main()
