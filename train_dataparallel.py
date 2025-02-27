import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
# 訓練用と評価用のデータローダーを作成する関数
def get_data_loaders(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    
    return train_loader, test_loader

# 学習関数
# モデルをトレーニングするための関数
def train(model, train_loader, optimizer, criterion, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
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

# メイン関数
# 学習の設定と実行を行うメイン関数
def main():
    # ハイパーパラメータ
    input_size = 784  # 28x28を平坦化
    hidden1_size = 128
    hidden2_size = 64
    output_size = 10
    batch_size = 16  # マルチGPU用に増加したバッチサイズ
    learning_rate = 0.001
    num_epochs = 5
    model_save_path = 'mnist_model_dataparallel.pth'

    # デバイスの設定とcuDNNベンチマーキングの有効化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # モデル、オプティマイザ、損失関数の作成
    model = SimpleMLP(input_size, hidden1_size, hidden2_size, output_size)
    
    # 複数のGPUが利用可能な場合はDataParallelを使用
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUを使用します!")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # データローダーの取得
    train_loader, test_loader = get_data_loaders(batch_size)

    # 学習時間の計測
    start_time = time.time()

    # モデルの学習
    model = train(model, train_loader, optimizer, criterion, device, num_epochs)

    # 学習時間の計測
    end_time = time.time()
    print(f"学習時間: {end_time - start_time:.2f} 秒")
    
    # テストデータでの評価
    accuracy = test(model, test_loader, device)
    
    # モデルの保存
    # DataParallelでラップされている場合はモジュールを取得
    if isinstance(model, nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model
        
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
    torch.save(model_info, 'model_info_dataparallel.pth')
    print(f"モデル情報を model_info_dataparallel.pth に保存しました")

if __name__ == "__main__":
    main()