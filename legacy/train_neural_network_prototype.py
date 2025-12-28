import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from agents.neural_player import PokerMLP, HandEncoder
from agents.montecarlo import setup_ai as setup_monte_carlo
from agents.random_player import setup_ai as setup_random
from game.game import setup_config, start_poker
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report


class PokerDataset(Dataset):
    """德州撲克訓練資料集"""

    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.FloatTensor(item['features'])
        action = torch.LongTensor([item['action']])
        return features, action.squeeze()


class DataCollector:
    """數據收集器，用於記錄expert的行為"""

    def __init__(self, expert_agent):
        self.expert_agent = expert_agent
        self.encoder = HandEncoder()
        self.collected_data = []

    def collect_game_data(self, num_games=100):
        """收集多場遊戲的專家行為數據"""
        print(f"開始收集 {num_games} 場遊戲的訓練數據...")

        for game_idx in range(num_games):
            if game_idx % 10 == 0:
                print(f"已完成 {game_idx}/{num_games} 場遊戲")

            # 設定遊戲參數
            config = setup_config(
                max_round=20, initial_stack=1000, small_blind_amount=5)
            config.register_player(
                name="expert", algorithm=self.expert_agent())
            config.register_player(name="random", algorithm=setup_random())

            # 執行遊戲並收集數據
            self._collect_single_game(config)

        print(f"數據收集完成！總共收集了 {len(self.collected_data)} 個決策樣本")
        return self.collected_data

    def _collect_single_game(self, config):
        """收集單場遊戲的數據"""
        # 這裡我們需要修改遊戲執行流程來收集中間狀態
        # 為了簡化，我們先創建一個mockup版本

        # 模擬一些訓練數據
        for _ in range(np.random.randint(5, 15)):  # 每場遊戲5-15個決策點
            # 隨機生成手牌
            hole_cards = [
                f"{np.random.choice(['C', 'D', 'H', 'S'])}{np.random.choice(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'])}" for _ in range(2)]

            # 隨機生成社區牌
            num_community = np.random.choice([0, 3, 4, 5])
            community_cards = [
                f"{np.random.choice(['C', 'D', 'H', 'S'])}{np.random.choice(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'])}" for _ in range(num_community)]

            # 隨機生成遊戲狀態
            round_count = np.random.randint(1, 21)
            my_stack = np.random.randint(100, 1500)
            opponent_stack = np.random.randint(100, 1500)
            pot_size = np.random.randint(10, 200)

            # 編碼特徵
            features = self.encoder.encode_game_state(
                hole_cards, community_cards, round_count,
                my_stack, opponent_stack, pot_size
            )

            # 使用expert策略決定動作（這裡用簡單的啟發式）
            if num_community == 0:  # preflop
                action = self._expert_preflop_decision(hole_cards)
            else:  # postflop
                action = self._expert_postflop_decision(
                    hole_cards, community_cards)

            self.collected_data.append({
                'features': features.tolist(),
                'action': action
            })

    def _expert_preflop_decision(self, hole_cards):
        """專家的preflop決策邏輯"""
        ranks = [card[1] for card in hole_cards]
        strong_ranks = ['A', 'K', 'Q', 'J']
        strong_count = sum(1 for rank in ranks if rank in strong_ranks)

        if strong_count == 2:
            return 2  # raise
        elif strong_count == 1:
            # mostly call, sometimes raise
            return np.random.choice([1, 2], p=[0.7, 0.3])
        else:
            # mostly fold, sometimes call
            return np.random.choice([0, 1], p=[0.8, 0.2])

    def _expert_postflop_decision(self, hole_cards, community_cards):
        """專家的postflop決策邏輯"""
        # 簡化的postflop邏輯
        all_cards = hole_cards + community_cards
        ranks = [card[1] for card in all_cards]

        # 檢查是否有對子
        if len(set(ranks)) < len(ranks):
            # more aggressive with pairs
            return np.random.choice([1, 2], p=[0.4, 0.6])
        else:
            # more conservative
            return np.random.choice([0, 1, 2], p=[0.4, 0.5, 0.1])

    def save_data(self, filename="poker_training_data.json"):
        """儲存收集的數據"""
        # 轉換numpy類型為Python原生類型
        serializable_data = []
        for item in self.collected_data:
            serializable_item = {
                # 轉換為Python float
                'features': [float(x) for x in item['features']],
                'action': int(item['action'])  # 轉換為Python int
            }
            serializable_data.append(serializable_item)

        with open(filename, 'w') as f:
            json.dump(serializable_data, f)
        print(f"訓練數據已儲存到 {filename}")


def train_model(data_file, model_save_path="agents/poker_model.pth", epochs=100):
    """訓練神經網路模型"""

    # 載入數據
    dataset = PokerDataset(data_file)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    model = PokerMLP(input_size=127, hidden_sizes=[
                     256, 128, 64], output_size=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 訓練歷史
    train_losses = []
    val_accuracies = []

    print("開始訓練神經網路...")

    for epoch in range(epochs):
        # 訓練階段
        model.train()
        total_train_loss = 0

        for features, actions in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # 驗證階段
        model.eval()
        val_predictions = []
        val_true = []

        with torch.no_grad():
            for features, actions in val_loader:
                outputs = model(features)
                predictions = torch.argmax(outputs, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_true.extend(actions.cpu().numpy())

        val_accuracy = accuracy_score(val_true, val_predictions)
        avg_train_loss = total_train_loss / len(train_loader)

        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accuracy)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # 儲存模型
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已儲存到 {model_save_path}")

    # 顯示最終結果
    print("\n=== 訓練完成 ===")
    print(f"最終驗證準確率: {val_accuracies[-1]:.4f}")
    print("\n分類報告:")
    print(classification_report(val_true, val_predictions,
          target_names=['Fold', 'Call', 'Raise']))

    # 繪製訓練曲線
    plt.figure(figsize=(15, 5))

    # 訓練損失
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 驗證準確率
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, 'r-', linewidth=2, label='Validation Accuracy')
    plt.title('Validation Accuracy Over Epochs',
              fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 損失和準確率對比
    plt.subplot(1, 3, 3)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    line1 = ax1.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
    line2 = ax2.plot(val_accuracies, 'r-', linewidth=2,
                     label='Validation Accuracy')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='b')
    ax2.set_ylabel('Validation Accuracy', color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')

    # 組合圖例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 輸出訓練曲線數據（額外的文字輸出）
    print("\n=== 訓練曲線數據 ===")
    print("訓練損失 (每10個epoch):")
    for i in range(0, len(train_losses), 10):
        print(f"Epoch {i}: {train_losses[i]:.4f}")

    print("\n驗證準確率 (每10個epoch):")
    for i in range(0, len(val_accuracies), 10):
        print(f"Epoch {i}: {val_accuracies[i]:.4f}")

    # 保存訓練歷史到JSON檔案
    training_history = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }
    with open('training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    print("\n訓練歷史已保存到 training_history.json")
    print("訓練曲線圖表已保存到 training_curves.png")

    return model


def main():
    """主要訓練流程"""

    print("=== 德州撲克神經網路訓練 ===")

    # 1. 收集訓練數據
    collector = DataCollector(setup_monte_carlo)  # 使用Monte Carlo作為expert
    collector.collect_game_data(num_games=200)  # 收集200場遊戲的數據
    collector.save_data("poker_training_data.json")

    # 2. 訓練模型
    model = train_model("poker_training_data.json", epochs=150)

    print("\n=== 訓練完成 ===")
    print("現在您可以使用 neural_player.py 來測試訓練好的模型！")


if __name__ == "__main__":
    main()
