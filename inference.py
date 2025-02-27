import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# 画像保存用のディレクトリを作成
def create_output_dir(output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# MLPモデルの定義（train.pyと同じ構造を維持）
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

# モデルをロードする関数
def load_model(model_path='mnist_model.pth', model_info_path='model_info.pth'):
    # モデル情報をロード
    model_info = torch.load(model_info_path)
    
    # モデルの構築
    model = SimpleMLP(
        model_info['input_size'],
        model_info['hidden1_size'],
        model_info['hidden2_size'],
        model_info['output_size']
    )
    
    # 学習済みパラメータをロード
    model.load_state_dict(torch.load(model_path))
    
    # 評価モードに設定
    model.eval()
    
    print(f"モデルをロードしました（精度: {model_info['accuracy']:.2f}%）")
    return model

# MNISTテストデータセットから画像を取得する関数
def get_test_images(num_images=5):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    images = []
    labels = []
    
    # ランダムにnum_images枚の画像を選択
    indices = torch.randperm(len(test_dataset))[:num_images]
    for idx in indices:
        image, label = test_dataset[idx]
        images.append(image)
        labels.append(label)
    
    return images, labels

# 画像を表示して保存する関数
def display_image(image, title=None, output_dir='output', filename=None, show=True):
    if isinstance(image, torch.Tensor):
        # 正規化を元に戻す
        image = image.squeeze().numpy()
        image = image * 0.3081 + 0.1307
        # 値の範囲を[0, 1]に調整
        image = np.clip(image, 0, 1)
    
    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    if title:
        plt.title(title)
    
    # 画像を保存
    if filename:
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    
    # 表示するかどうか
    if show:
        plt.show()
    else:
        plt.close()

# 推論を実行する関数
def predict(model, image, device='cpu'):
    # デバイスの設定
    device = torch.device(device)
    model = model.to(device)
    
    # 画像の前処理
    if not isinstance(image, torch.Tensor):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image = transform(image)
    
    # バッチ次元を追加して推論
    image = image.unsqueeze(0).to(device)
    image = image.view(image.size(0), -1)  # 平坦化
    
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

# 確率分布のグラフを表示して保存する関数
def plot_probabilities(probabilities, output_dir='output', filename=None, show=True):
    plt.figure(figsize=(8, 3))
    plt.bar(range(10), probabilities)
    plt.xlabel('数字')
    plt.ylabel('確率')
    plt.title('予測確率分布')
    plt.xticks(range(10))
    
    # グラフを保存
    if filename:
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    
    # 表示するかどうか
    if show:
        plt.show()
    else:
        plt.close()

# メイン関数
def main(show_plots=True, save_plots=True):
    # 出力ディレクトリの作成
    output_dir = create_output_dir()
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # モデルのロード
    model = load_model()
    model = model.to(device)
    
    # テスト画像の取得
    images, true_labels = get_test_images(5)
    
    # 各画像に対して推論を実行
    for i, (image, true_label) in enumerate(zip(images, true_labels)):
        # 推論
        predicted_class, confidence, probabilities = predict(model, image, device)
        
        # 結果の表示
        print(f"画像 {i+1}:")
        print(f"  正解ラベル: {true_label}")
        print(f"  予測クラス: {predicted_class}")
        print(f"  信頼度: {confidence:.4f}")
        
        # 画像の表示と保存
        title = f"True: {true_label}, Pred: {predicted_class}"
        image_filename = f"image_{i+1}_true_{true_label}_pred_{predicted_class}.png" if save_plots else None
        display_image(image, title, output_dir, image_filename, show=show_plots)
        
        # 確率分布のグラフ表示と保存
        prob_filename = f"probs_{i+1}_true_{true_label}_pred_{predicted_class}.png" if save_plots else None
        plot_probabilities(probabilities, output_dir, prob_filename, show=show_plots)
        
        print("-" * 50)
    
    print(f"画像は {output_dir} ディレクトリに保存されました。")

# カスタム画像で推論を実行する関数
def predict_custom_image(image_path, model=None, show_plots=True, save_plots=True):
    # 出力ディレクトリの作成
    output_dir = create_output_dir()
    
    if model is None:
        model = load_model()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 画像の読み込みと前処理
    image = Image.open(image_path).convert('L')  # グレースケールに変換
    image = image.resize((28, 28))  # MNISTと同じサイズにリサイズ
    
    # 画像のファイル名（パスから抽出）
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 画像の表示と保存
    image_filename = f"custom_{image_name}_input.png" if save_plots else None
    display_image(np.array(image), "入力画像", output_dir, image_filename, show=show_plots)
    
    # 推論
    predicted_class, confidence, probabilities = predict(model, image, device)
    
    # 結果の表示
    print(f"予測クラス: {predicted_class}")
    print(f"信頼度: {confidence:.4f}")
    
    # 確率分布のグラフ表示と保存
    prob_filename = f"custom_{image_name}_probs_pred_{predicted_class}.png" if save_plots else None
    plot_probabilities(probabilities, output_dir, prob_filename, show=show_plots)
    
    print(f"画像は {output_dir} ディレクトリに保存されました。")
    
    return predicted_class, confidence

if __name__ == "__main__":
    # 画像を表示するかどうか
    show_plots = True
    # 画像を保存するかどうか
    save_plots = True
    
    main(show_plots=show_plots, save_plots=save_plots)
    
    # カスタム画像がある場合はコメントを外して実行
    # predict_custom_image("path/to/your/image.png", show_plots=show_plots, save_plots=save_plots) 