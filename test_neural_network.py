import json
from game.game import setup_config, start_poker
from agents.neural_player import setup_ai as setup_neural
from agents.montecarlo import setup_ai as setup_monte_carlo
from agents.random_player import setup_ai as setup_random
from agents.rule_base import setup_ai as setup_rule_base
import numpy as np
import matplotlib.pyplot as plt


def test_agent_performance(agent_name, agent_setup, num_games=50):
    """測試單個代理的性能"""
    wins = 0
    total_winnings = 0
    game_results = []

    print(f"\n測試 {agent_name} 對戰隨機玩家...")

    for game_idx in range(num_games):
        if game_idx % 10 == 0:
            print(f"進度: {game_idx}/{num_games}")

        # 設定遊戲
        config = setup_config(
            max_round=20, initial_stack=1000, small_blind_amount=5)
        config.register_player(name="agent", algorithm=agent_setup())
        config.register_player(name="random", algorithm=setup_random())

        # 執行遊戲
        result = start_poker(config, verbose=0)  # 不顯示詳細過程

        # 分析結果
        agent_stack = None
        random_stack = None

        for player in result["players"]:
            if player["name"] == "agent":
                agent_stack = player["stack"]
            elif player["name"] == "random":
                random_stack = player["stack"]

        # 計算勝負
        if agent_stack > random_stack:
            wins += 1

        winnings = agent_stack - 1000  # 初始籌碼是1000
        total_winnings += winnings
        game_results.append(winnings)

    win_rate = wins / num_games
    avg_winnings = total_winnings / num_games

    return {
        'agent_name': agent_name,
        'win_rate': win_rate,
        'avg_winnings': avg_winnings,
        'total_winnings': total_winnings,
        'game_results': game_results
    }


def compare_agents(num_games=50):
    """比較不同AI代理的性能"""

    agents = [
        ("Neural Network", setup_neural),
        ("Monte Carlo", setup_monte_carlo),
        ("Rule Base", setup_rule_base),
        ("Random", setup_random)
    ]

    results = []

    print("=== AI代理性能比較 ===")

    for agent_name, agent_setup in agents:
        try:
            result = test_agent_performance(agent_name, agent_setup, num_games)
            results.append(result)
        except Exception as e:
            print(f"測試 {agent_name} 時發生錯誤: {e}")
            continue

    # 顯示結果
    print("\n=== 測試結果 ===")
    print(f"{'代理名稱':<15} {'勝率':<10} {'平均獲利':<12} {'總獲利':<10}")
    print("-" * 50)

    for result in results:
        print(f"{result['agent_name']:<15} {result['win_rate']:<10.3f} {result['avg_winnings']:<12.1f} {result['total_winnings']:<10.1f}")

    # 繪製結果圖表
    plot_comparison_results(results)

    return results


def plot_comparison_results(results):
    """繪製比較結果圖表"""

    if not results:
        print("沒有結果可以繪製")
        return

    # 設置中文字體（如果需要）
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS',
                                       'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    agent_names = [r['agent_name'] for r in results]
    win_rates = [r['win_rate'] for r in results]
    avg_winnings = [r['avg_winnings'] for r in results]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    # 1. 勝率比較
    bars1 = ax1.bar(agent_names, win_rates,
                    color=colors[:len(agent_names)], alpha=0.8)
    ax1.set_title('Win Rate Comparison', fontsize=16,
                  fontweight='bold', pad=20)
    ax1.set_ylabel('Win Rate', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    # 在柱狀圖上添加數值標籤
    for i, v in enumerate(win_rates):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center',
                 va='bottom', fontweight='bold')

    # 2. 平均獲利比較
    bars2 = ax2.bar(agent_names, avg_winnings,
                    color=colors[:len(agent_names)], alpha=0.8)
    ax2.set_title('Average Winnings Comparison',
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Average Winnings', fontsize=12)
    ax2.grid(True, alpha=0.3)
    # 在柱狀圖上添加數值標籤
    for i, v in enumerate(avg_winnings):
        ax2.text(i, v + (max(avg_winnings) * 0.02),
                 f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    # 3. 累計獲利趨勢
    ax3.set_title('Cumulative Winnings Over Games',
                  fontsize=16, fontweight='bold', pad=20)
    ax3.set_xlabel('Game Number', fontsize=12)
    ax3.set_ylabel('Cumulative Winnings', fontsize=12)

    for i, result in enumerate(results):
        cumulative = np.cumsum(result['game_results'])
        ax3.plot(range(1, len(cumulative) + 1), cumulative,
                 label=result['agent_name'], linewidth=3, color=colors[i % len(colors)])

    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. 獲利分佈直方圖
    ax4.set_title('Winnings Distribution', fontsize=16,
                  fontweight='bold', pad=20)
    ax4.set_xlabel('Winnings per Game', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)

    for i, result in enumerate(results):
        ax4.hist(result['game_results'], bins=20, alpha=0.7,
                 label=result['agent_name'], color=colors[i % len(colors)])

    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('agent_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 額外的文字輸出（保留原有功能）
    print("\n=== 詳細比較結果 ===")

    # 1. 勝率比較
    print("\n1. 勝率比較:")
    for result in results:
        print(f"  {result['agent_name']:<15}: {result['win_rate']:.3f}")

    # 2. 平均獲利比較
    print("\n2. 平均獲利比較:")
    for result in results:
        print(f"  {result['agent_name']:<15}: {result['avg_winnings']:.1f}")

    # 3. 累計獲利趨勢分析
    print("\n3. 累計獲利趨勢分析:")
    for result in results:
        cumulative = np.cumsum(result['game_results'])
        final_cumulative = cumulative[-1]
        max_cumulative = np.max(cumulative)
        min_cumulative = np.min(cumulative)
        print(
            f"  {result['agent_name']:<15}: 最終={final_cumulative:.1f}, 最高={max_cumulative:.1f}, 最低={min_cumulative:.1f}")

    # 4. 獲利分佈統計
    print("\n4. 獲利分佈統計:")
    for result in results:
        game_results = np.array(result['game_results'])
        mean_winnings = np.mean(game_results)
        std_winnings = np.std(game_results)
        positive_games = np.sum(game_results > 0)
        total_games = len(game_results)
        print(
            f"  {result['agent_name']:<15}: 平均={mean_winnings:.1f}, 標準差={std_winnings:.1f}, 正獲利局數={positive_games}/{total_games}")

    # 保存詳細結果到JSON
    detailed_results = {
        'summary': results,
        'analysis': {
            'best_win_rate': max(results, key=lambda x: x['win_rate']),
            'best_avg_winnings': max(results, key=lambda x: x['avg_winnings']),
            'most_consistent': min(results, key=lambda x: np.std(x['game_results']))
        }
    }

    with open('detailed_comparison_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2, default=lambda x: float(
            x) if isinstance(x, np.floating) else x)

    print("\n詳細結果已保存到 detailed_comparison_results.json")
    print("比較圖表已保存到 agent_comparison_results.png")


def tournament_test():
    """錦標賽測試：所有AI互相對戰"""

    agents = [
        ("Neural Network", setup_neural),
        ("Monte Carlo", setup_monte_carlo),
        ("Rule Base", setup_rule_base)
    ]

    results_matrix = np.zeros((len(agents), len(agents)))

    print("\n=== 錦標賽測試 ===")

    for i, (agent1_name, agent1_setup) in enumerate(agents):
        for j, (agent2_name, agent2_setup) in enumerate(agents):
            if i == j:
                continue

            print(f"{agent1_name} vs {agent2_name}")

            wins = 0
            num_games = 20

            for _ in range(num_games):
                config = setup_config(
                    max_round=20, initial_stack=1000, small_blind_amount=5)
                config.register_player(
                    name="player1", algorithm=agent1_setup())
                config.register_player(
                    name="player2", algorithm=agent2_setup())

                result = start_poker(config, verbose=0)

                player1_stack = None
                player2_stack = None

                for player in result["players"]:
                    if player["name"] == "player1":
                        player1_stack = player["stack"]
                    elif player["name"] == "player2":
                        player2_stack = player["stack"]

                if player1_stack > player2_stack:
                    wins += 1

            win_rate = wins / num_games
            results_matrix[i][j] = win_rate
            print(f"  {agent1_name} 勝率: {win_rate:.3f}")

    # 顯示結果矩陣
    print("\n=== 錦標賽結果矩陣 ===")
    print("        ", end="")
    for agent_name, _ in agents:
        print(f"{agent_name[:10]:<12}", end="")
    print()

    for i, (agent_name, _) in enumerate(agents):
        print(f"{agent_name[:10]:<8}", end="")
        for j in range(len(agents)):
            if i == j:
                print("    -    ", end="")
            else:
                print(f"  {results_matrix[i][j]:.3f}  ", end="")
        print()


def main():
    """主要測試流程"""

    print("=== 神經網路AI測試程式 ===")

    # 1. 基本性能測試
    print("\n1. 執行基本性能測試...")
    comparison_results = compare_agents(num_games=30)

    # 2. 錦標賽測試
    print("\n2. 執行錦標賽測試...")
    tournament_test()

    # 3. 保存結果
    print("\n3. 保存測試結果...")
    with open('test_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)

    print("\n=== 測試完成 ===")
    print("結果已保存到 test_results.json 和 agent_comparison_results.png")
    print("詳細分析已保存到 detailed_comparison_results.json")


if __name__ == "__main__":
    main()
