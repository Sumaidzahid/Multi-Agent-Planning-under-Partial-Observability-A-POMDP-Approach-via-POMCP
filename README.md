# Multi-Agent-POMDP-via-POMCP
This Project presents a decentralized navigation framework for multi-agent systems operating in partially observable grid environments. Each agent independently models its decision process as a Partially Observable Markov Decision Process (POMDP) and employs Partially Observable Monte Carlo Planning (POMCP) for online, real-time action selection.


Note: main.py Runs the whole project 

Suggestions: Create a Virtual python envoirment before running this program

## Grid MAP 

<img width="800" height="800" alt="map_grid" src="https://github.com/user-attachments/assets/4c145c51-aacc-4785-87ac-07324f28e48f" />

## System Architecture

framework where each agent maintains its own belief
state, performs local planning using Partially Observable
Monte Carlo Planning (POMCP) and updates local actionobservation
based on the planned outcome. The architecture
integrates four main components:
• Environment and State Management
• Belief State Management
• Local POMCP-Based Planning
• Multi-Agent Execution
This design ensures modularity, scalability, and robustness
under partial observability, uncertainty and independent
planning.

## Architectural Overview

Using a decentralized approach, each agent manages its own
belief state and runs its own planning process without any
communication with other agents. The system environment
takes all agent’s chosen action as joint actions and executes
them providing each agent with its own reward and observations.
This setup avoids the high computational cost of
centralized planning. The Fig. 2 shows the Execution cycle
and Data Flow in Decentralized POMCP Planning.

<img width="812" height="479" alt="image" src="https://github.com/user-attachments/assets/86832c0c-62e5-42a1-8650-852aaf780d92" />

Agents plan their next action at each step based on their
belief state. The environment then executes all selected actions
at the same time. The environment then returns the observations
which are used to update each agent’s belief and
improve future planning.

## Results 

- 4/5 episodes successful (80% success rate)
- Agent 1 achieved goal in all 5 episodes (100% success), average cumulative reward: 3,214, average steps: 197
- Agent 2 achieved goal in 4/5 episodes (80% success), average cumulative reward: 3,854, average steps: 197
- Episode 3 only failure — Agent 2 exhibited prolonged exploratory behavior, maximum 499 steps reached
- Episode 5 highest efficiency: 68.73 total efficiency score, both agents reached goal in 186 steps
- Final test run: both agents reached goal in 66 steps (Agent 1 reward: 57, Agent 2 reward: 3,243)
- Planning horizon: 3 | Simulations per step: 22 | γ_POMCP: 0.99 | γ_shape: 0.90
- Environment: 20×20 grid | 2 agents | Max steps: 500 | Observation radius: 2
