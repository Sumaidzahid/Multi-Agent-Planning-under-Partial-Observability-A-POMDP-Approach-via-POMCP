import numpy as np
from BeliefStateManager import BeliefManager
from scipy.stats import entropy

def belief_entropy(belief_map):
    eps = 1e-6
    p = np.clip(belief_map, eps, 1 - eps)
    return entropy([p.mean(), 1 - p.mean()])

def manhattan(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def manh_X(pos1, pos2):
    return abs(pos1[0] - pos2[0])

def manh_Y(pos1, pos2):
    return abs(pos1[1] - pos2[1])


class StateManager:

    def __init__(self, true_map, start, goal, belief_mgr: BeliefManager):
        self.true_map = true_map  # Real grid (unknown to the agent)
        self.H, self.W = true_map.shape
        self.belief_mgr = belief_mgr
        self.visited = {agent_id: set([pos]) for agent_id, pos in start.items()}  # initialize with start pos
        self.agent_pos ={agent_id: pos for agent_id, pos in start.items() }
        self.goal_pos ={goal_id: pos for goal_id, pos in goal.items() }
        self.best_distance = {agent_id: manhattan(start[agent_id], goal[agent_id]) for agent_id in start.keys()}

        self.success_prob = 1.0
        self.fail_prob = 0.0

    # -------------------------------------------------------
    # 1. OBSERVATION MODEL(what the agent can see)
    # -------------------------------------------------------
    def observation(self, agent_id, position, map_grid, belief_map, H, W, radius=2):
        """
        Agent observes cells in a given radius (usually 1 step around).
        Returns a hashable tuple for POMCP.
        """

        row, col = position
        observed_cells = {}
        for drow in range(H):
            for dcol in range(W):
                dist = abs(drow - row) + abs(dcol - col)
                if dist <= radius: 
                    observed_cells[(drow, dcol)] = int(map_grid[drow, dcol]) 
                else: 
                    observed_cells[(drow, dcol)] = int(belief_map[drow, dcol])
        goal = self.goal_pos[agent_id]
        dx = np.sign(goal[0] - row)
        dy = np.sign(goal[1] - col)
        observed_cells[("goal_dir",)] = (int(dx), int(dy))

        return tuple(sorted(observed_cells.items(), key=lambda x: str(x[0])))

    
    # -------------------------------------------------------
    # 2. TRANSITION MODEL (movement uncertainty)
    # -------------------------------------------------------
    def transition_model(self, actions: dict, particle_state=None, use_belief=False):
        new_positions = {}

        if use_belief:
            agent_positions = particle_state['agent_pos']
            map_grid = particle_state['map']
        else:
            agent_positions = self.agent_pos
            map_grid = self.true_map

        for agent_id, action in actions.items():
            row, col = agent_positions[agent_id]

            if action == "up":
                nrow, ncol = row - 1, col
            elif action == "down":
                nrow, ncol = row + 1, col
            elif action == "left":
                nrow, ncol = row, col - 1
            elif action == "right":
                nrow, ncol = row, col + 1
            else:
                nrow, ncol = row, col

            if not (0 <= nrow < self.H and 0 <= ncol < self.W) \
                or map_grid[nrow, ncol] == 1:
                nrow, ncol = row, col

            new_positions[agent_id] = (nrow, ncol)

        return new_positions
    
    def single_agent_transition(self, agent_id, state, action):
        """
        Deterministic transition for a *simulated* single agent inside POMCP.
        Uses the state's own position and map, not the real environment state.
        """
        row, col = state["pos"]
        map_grid = state["map"]  

        if action == "up":
            nrow, ncol = row - 1, col
        elif action == "down":
            nrow, ncol = row + 1, col
        elif action == "left":
            nrow, ncol = row, col - 1
        elif action == "right":
            nrow, ncol = row, col + 1
        else:  # 'stay' or anything else
            nrow, ncol = row, col

        # Check bounds and obstacles in the *particle map*
        if not (0 <= nrow < self.H and 0 <= ncol < self.W) or map_grid[nrow, ncol] == 1:
            nrow, ncol = row, col

        return {
            "pos": (nrow, ncol),
            "map": map_grid
        }
    
    # -------------------------------------------------------
    # 3. REWARD FUNCTION
    # -------------------------------------------------------

    def reward(self, agent_id, state, action, next_state, belief, next_belief):

        # 1. Environment reward
        old_d = manhattan(state['agent_pos'][agent_id], self.goal_pos[agent_id])
        new_d = manhattan(next_state['agent_pos'][agent_id], self.goal_pos[agent_id])

        # old_dx = manh_X(state['agent_pos'][agent_id], self.goal_pos[agent_id])
        # new_dx = manh_X(next_state['agent_pos'][agent_id], self.goal_pos[agent_id])

        # old_dy = manh_Y(state['agent_pos'][agent_id], self.goal_pos[agent_id])
        # new_dy = manh_Y(next_state['agent_pos'][agent_id], self.goal_pos[agent_id])


        r_env = -0.1  # time penalty

        # if old_dx > new_dx:
        #     r_env += 1
        # if old_dy > new_dy:
        #     r_env += 1
        # if old_dx < new_dx:
        #     r_env -= 5
        # if old_dy < new_dy:
        #     r_env -= 5
             
        # if new_d < old_d:
        #     r_env += 1 # progress reward
        # if new_d > old_d:
        #     r_env -= 0.5

        if next_state['agent_pos'][agent_id] == self.goal_pos[agent_id]:
            r_env += 100

        if action != 'stay' and next_state['agent_pos'][agent_id] == state['agent_pos'][agent_id]:
            r_env -= 1  # wall penalty

        # Revisit penalty: penalize revisiting already visited cells
        if next_state['agent_pos'][agent_id] in self.visited[agent_id]:
            r_env -= 0.5  # small penalty for revisiting

        if action == 'stay':
            r_env -= 0.5

        # 2. PBRS shaping
        discount_factor = 0.90
        Phi_old = -1/( 1 + old_d)
        Phi_new = -1/( 1 +  new_d)

        r_pbrs =  (discount_factor * Phi_new - Phi_old)

        # 3. Information gain (increased weight for exploration)
        old_entropy = belief_entropy(belief[agent_id])
        new_entropy = belief_entropy(next_belief[agent_id])
        r_info = (old_entropy - new_entropy)
        r_info = max(-3, min(3, r_info))
        r_info = np.clip(r_info, -1.0, 1.0)

        return r_env + r_pbrs + r_info


    # -------------------------------------------------------
    # 4. APPLY ACTIONS TO ALL AGENTS
    # -------------------------------------------------------
        

    def apply_actions(self, action_dict):
        """
        actions_dict = {agent_id: action}
        Returns:
            new_positions
            true_observations
            rewards
        """

        new_positions = {}
        observations = {}
        rewards = {}

        # Current state
        state = {'agent_pos': self.agent_pos.copy()}
        old_belief = {
        aid: self.belief_mgr.belief[aid].copy()
        for aid in action_dict.keys()
        }

        # Apply actions
        new_positions = self.transition_model(action_dict)  # joint transition

        # Next state
        next_state = {'agent_pos': new_positions.copy()}

        # Get observations and update beliefs
        for agent_id in action_dict.keys():
            obs = self.observation(agent_id, new_positions[agent_id], self.true_map, self.belief_mgr.belief[agent_id], self.H, self.W, radius=2)
            observations[agent_id]  = obs
            self.belief_mgr.update_belief(agent_id, obs)

        next_belief = self.belief_mgr.belief.copy()

        # Update positions
        for agent_id, new_pos in new_positions.items():
            self.agent_pos[agent_id] = new_pos
            self.visited[agent_id].add(new_pos)  # mark as visited
            action = action_dict[agent_id]
            rewards[agent_id] = self.reward(agent_id, state, action, next_state, old_belief, next_belief)

        return new_positions, observations, rewards

    # -------------------------------------------------------
    # 5. All Agents at Goal
    # -------------------------------------------------------

    def all_agents_at_goal(self):
        for aid, pos in self.agent_pos.items():
            if pos != self.goal_pos[aid]:
                return False
        return True

        

