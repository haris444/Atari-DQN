import argparse

# Default hyperparameters
GAMMA = 0.99  # Discount factor for Bellman equation
EPS_START = 1  # Initial exploration rate
EPS_END = 0.05  # Final exploration rate 
EPS_DECAY = 25000  # Exploration decay rate
WARMUP = 10000  # Steps before starting updates
DEFAULT_SEED = 42  # Default random seed

def parse_args():
    """Parse command line arguments and return args object"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="breakout", type=str, choices=["pong", "breakout", "boxing","spaceinvaders","pacman"], help="env name")
    parser.add_argument('--model', default="dqn", type=str, choices=["dqn", "dueldqn"], help="dqn model")
    parser.add_argument('--gpu', default=0, type=int, help="which gpu to use")
    parser.add_argument('--algorithm', default="dqn", type=str, choices=["dqn", "expected_sarsa"], help="RL algorithm")
    parser.add_argument('--weighted-is', action='store_true', help="Use weighted importance sampling")
    parser.add_argument('--lr', default=2.5e-4, type=float, help="learning rate")
    parser.add_argument('--epoch', default=10001, type=int, help="training epoch")
    parser.add_argument('--batch-size', default=32, type=int, help="batch size")
    parser.add_argument('--target-update-episodes', default=10, type=int, help="Update target network every X episodes")
    parser.add_argument('--ddqn', action='store_true', help="double dqn/dueldqn")
    parser.add_argument('--eval-cycle', default=500, type=int, help="evaluation cycle")
    parser.add_argument('--save-cycle', default=10, type=int, help="save model every X evaluations")
    parser.add_argument('--min-replay-size', default=10000, type=int, help="Minimum replay memory size before training")
    parser.add_argument('--clip-weights', action='store_true', help="Clip importance sampling weights")
    parser.add_argument('--max-weight', default=10.0, type=float, help="Maximum importance sampling weight when clipping")
    parser.add_argument('--log-dir', type=str, help="Directory to save logs and checkpoints (overrides default)")
    parser.add_argument('--seed', default=DEFAULT_SEED, type=int, help="Random seed for reproducibility")
    parser.add_argument('--deterministic', action='store_true', help="Enable deterministic behavior in PyTorch")
    
    return parser.parse_args()