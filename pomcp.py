from Tree import TreeBuilder, UCB
import numpy as np
from BeliefStateManager import BeliefManager as BM
from State_Manager import StateManager as SM


class POMCPAgent:
    def __init__(self, agent_id, state_mgr: SM,
                 belief_mgr: BM, gamma=0.95, horizon=10):
        self.agent_id = agent_id
        self.state_mgr = state_mgr
        self.belief_mgr = belief_mgr
        self.gamma = gamma
        self.horizon = horizon
        self.tree = TreeBuilder()
        self.actions = ["up", "down", "left", "right", "stay"]

    def bestAction(self, n_simulations=100):
        """
        Run POMCP for this agent only.
        Returns: best local action for this agent.
        """
        for _ in range(n_simulations):
            # sample a map from this agent's belief
            map_sample = self._sample_map()
            state = {
                "pos": self.state_mgr.agent_pos[self.agent_id],
                "map": map_sample
            }
            belief = self.belief_mgr.belief[self.agent_id].copy()
            self._simulate(self.tree.root, state, belief, depth=0)

        # pick best action from root
        root_node = self.tree.nodes[self.tree.root]
        best_a, best_N = None, -1

        for a, a_hist in root_node["children"].items():
            a_node = self.tree.nodes[a_hist]
            if a_node["N"] > best_N:
                best_N = a_node["N"]
                best_a = a

        # fallback if no children yet
        if best_a is None:
            current_state = { "pos": self.state_mgr.agent_pos[self.agent_id], "map": self._sample_map() }
            best_a = self._greedy_goal_action(current_state)

        return best_a

    def _simulate(self, history, state, belief, depth):
        if depth >= self.horizon:
            return 0.0

        # expand history node if not in tree
        if history not in self.tree.nodes:
            self.tree.nodes[history] = {
                "parent": None,
                "children": {},
                "N": 0,
                "V": 0.0,
                "B": [],
                "is_action": False
            }
            return self._rollout(state, belief, depth)

        node = self.tree.nodes[history]
        node["N"] += 1

        # select action
        action = self._select_action(state, history, node)

        # action node
        a_hist = self.tree.getCreateActionNode(history, action)
        a_node = self.tree.nodes[a_hist]

        # transition
        next_state = self._transition(state, action)

        # observation (local)
        obs = self._observe(next_state, belief)

        # update belief locally
        next_belief = belief.copy()
        for key, v in obs:
            if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], int):
                r, c = key
                next_belief[r, c] = 0.1 if v == 0 else 0.9

        # observation node
        o_hist = self.tree.getCreateObservationNode(a_hist, obs)

        # reward (per-agent)
        r = self._reward(state, action, next_state, belief, next_belief)

        # recursive simulate
        G = r + self.gamma * self._simulate(o_hist, next_state, next_belief, depth + 1)

        # Backup observation node
        o_node = self.tree.nodes[o_hist]
        o_node["N"] += 1
        o_node["V"] += (G - o_node["V"]) / o_node["N"]


        # backup on action node
        a_node["N"] += 1
        a_node["V"] += (G - a_node["V"]) / a_node["N"]

        # Backup history node
        node["V"] += (G - node["V"]) / node["N"]

        return G

    def _greedy_goal_action(self, state):
        pos = state["pos"]
        goal = self.state_mgr.goal_pos[self.agent_id]
        best_a, best_d = None, float("inf")

        for a in self.actions:
            next_pos = self.state_mgr.single_agent_transition(self.agent_id, state, a)["pos"]
            d = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
            if d < best_d:best_d, best_actions = d, [a]
            elif d == best_d: best_actions.append(a)
        return np.random.choice(best_actions)

    def _rollout(self, state, belief, depth):
        if depth >= self.horizon:
            return 0.0
        # Action selection
        if np.random.rand() < 0.9:
            action = self._greedy_goal_action(state)
        else:
            action = np.random.choice(self.actions)

        next_state = self._transition(state, action)
        obs = self._observe(next_state, belief)

        # update local belief copy
        next_belief = belief.copy()
        for (r, c), v in obs:
            next_belief[r, c] = 0.1 if v == 0 else 0.9

        r = self._reward(state, action, next_state, belief, next_belief)
        return r + self.gamma * self._rollout(next_state, next_belief, depth + 1)


    def _transition(self, state, action):
        return self.state_mgr.single_agent_transition(self.agent_id, state, action)

    def _observe(self, state, belief):
        """
        Local observation for this agent using the sampled map.
        """
        pos = state["pos"]
        return self.state_mgr.observation(self.agent_id,pos,state["map"],belief,self.state_mgr.H,self.state_mgr.W,radius=2)

    def _reward(self, state, action, next_state, belief, next_belief):
        """
        Convert POMCP simulation state into the format expected by StateManager.reward().
        """

        return self.state_mgr.reward(
            self.agent_id,
            {"agent_pos": {self.agent_id: state["pos"]}},
            action,
            {"agent_pos": {self.agent_id: next_state["pos"]}},
            belief,
            next_belief)


    def _select_action(self, state, history, node):
        """
        Simple POMCP action selection with UCB over local actions.
        """
        children = node["children"]

        # explore untried actions first
        untried = [a for a in self.actions if a not in children]
        if untried:
            a = self._greedy_goal_action(state)
            if a in untried:
                return a
            
            return np.random.choice(untried)

        # otherwise use UCB
        N = node["N"]
        best_a, best_score = None, -float("inf")
        for a in self.actions:
            a_hist = children[a]
            a_node = self.tree.nodes[a_hist]
            score = UCB(N, a_node["N"], a_node["V"], c=3.0)
            if score > best_score:
                best_score = score
                best_a = a
        return best_a

    def _sample_map(self):
        """
        Sample a map particle from this agent's belief.
        """
        parts = self.belief_mgr.particles[self.agent_id]
        if len(parts) == 0:
            parts = self.belief_mgr.particle_sampling(self.agent_id) 
        return parts[np.random.randint(len(parts))]
    