# DQN CartPole v1 Results

This folder contains the results of the **DQN (Deep Q-Network)** training on the CartPole-v1 environment using PyTorch.

## Experiment Details

- **Environment**: OpenAI Gym `CartPole-v1`
- **Episodes per Run**: 300
- **Evaluation Metric**: Last 50 episode scores of each run
- **Repetitions**: 50 independent training runs
- **Device**: Apple MacBook (M3 Air)

## Aggregated Results

- **Average Min Score**: 34.58  
- **Average Max Score**: 359.34  
- **Average Mean Score**: 192.82  
- **Average Standard Deviation**: 70.60  

## Files

- `dqn_cartpole_v1_results.csv`: Summary of all 50 runs (min, max, average, std per run)
- `dqn_cartpole_v1_results.txt`: Formatted version of the same result (for human readability)

## Notes

The goal of this experiment was to establish a reproducible baseline for evaluating DQN performance consistency.  
These results will serve as a reference for future improvements such as Double DQN, Dueling DQN, etc.

