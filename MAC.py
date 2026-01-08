from Tree import TreeBuilder
from BeliefStateManager import BeliefManager as BM
from State_Manager import StateManager as SM
from pomcp import POMCPAgent

class MultiAgentController:
    def __init__(self, state_mgr:SM, belief_mgr:BM, agent_ids, gamma=0.95, horizon=10):
        self.state_mgr = state_mgr
        self.belief_mgr = belief_mgr
        self.agent_ids = agent_ids
        self.trails = {aid: [] for aid in agent_ids}

        self.agents = {
            aid: POMCPAgent(aid, state_mgr, belief_mgr, gamma, horizon)
            for aid in agent_ids}

        # local histories per agent
        self.histories = {aid: () for aid in agent_ids}


    def step(self, n_simulations=100):
        # 1. Each agent independently chooses its action
        joint_action = {}
        for aid, planner in self.agents.items():
            a = planner.bestAction(n_simulations)
            joint_action[aid] = a

        # 2. Apply joint action in the real environment
        new_pos, observations, rewards = self.state_mgr.apply_actions(joint_action)
        
        # Agent trails
        for aid in self.agent_ids:
            self.trails[aid].append(self.state_mgr.agent_pos[aid])

        # 3. Update beliefs from real observations
        for aid, obs in observations.items():
            self.belief_mgr.update_belief(aid, obs)
        # 4. Update each agent's local history and re-root its tree
        for aid in self.agent_ids:
            a = joint_action[aid]
            o = observations[aid]

            # extend local history
            old_hist = self.histories[aid]
            new_hist = old_hist + (a, o)
            self.histories[aid] = new_hist

            # re-root the agent's tree
            tree = self.agents[aid].tree
            if new_hist in tree.nodes:
                tree.make_root(new_hist)
            else:
                # if not in tree, reset to empty
                self.agents[aid].tree = TreeBuilder()

        return joint_action, observations, rewards
