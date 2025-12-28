import argparse
import json

from game.game import setup_config, start_poker


def parse_args():
    p = argparse.ArgumentParser(description="Play against the trained policy (models/policy_nn.pt).")
    p.add_argument("--opponent", default="baseline7", help="baseline1~7 | random | rule | human")
    p.add_argument("--max-round", type=int, default=20)
    p.add_argument("--stack", type=int, default=1000)
    p.add_argument("--sb", type=int, default=5)
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--model", default="models/policy_nn.pt", help="Path to policy weights")
    p.add_argument("--swap", action="store_true", help="Swap seats (policy becomes player2)")
    return p.parse_args()


def build_opponent(name: str):
    name = name.lower().strip()
    if name == "human":
        from agents.console_player import setup_ai as console_ai

        return "Human", console_ai()
    if name == "random":
        from agents.random_player import setup_ai as random_ai

        return "Random", random_ai()
    if name == "rule":
        from agents.rule_base import setup_ai as rule_ai

        return "RuleBase", rule_ai()
    if name.startswith("baseline"):
        n = int(name.replace("baseline", ""))
        if n == 1:
            from baseline1 import setup_ai as setup
        elif n == 2:
            from baseline2 import setup_ai as setup
        elif n == 3:
            from baseline3 import setup_ai as setup
        elif n == 4:
            from baseline4 import setup_ai as setup
        elif n == 5:
            from baseline5 import setup_ai as setup
        elif n == 6:
            from baseline6 import setup_ai as setup
        elif n == 7:
            from baseline7 import setup_ai as setup
        else:
            raise ValueError("baseline must be 1~7")
        return f"baseline{n}", setup()
    raise ValueError("Unknown opponent. Use baseline1~7 | random | rule | human")


def main():
    args = parse_args()

    # Policy player (loads models/policy_nn.pt)
    from agents.RLplayer import RLAgent

    policy = RLAgent(model_path=args.model)
    opp_name, opponent = build_opponent(args.opponent)

    config = setup_config(max_round=args.max_round, initial_stack=args.stack, small_blind_amount=args.sb)

    if args.swap:
        config.register_player(name=opp_name, algorithm=opponent)
        config.register_player(name="PolicyNN", algorithm=policy)
    else:
        config.register_player(name="PolicyNN", algorithm=policy)
        config.register_player(name=opp_name, algorithm=opponent)

    game_result = start_poker(config, verbose=args.verbose)
    print(json.dumps(game_result, indent=4))


if __name__ == "__main__":
    main()


