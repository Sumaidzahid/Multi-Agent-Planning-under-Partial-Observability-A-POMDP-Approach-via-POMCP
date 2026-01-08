import numpy as np

class BeliefManager:
    def __init__(self, true_map, agent_id):
        self.true_map = true_map
        self.H, self.W = true_map.shape

        self.belief = {
            aid: np.full((self.H, self.W), 0.5, dtype=float)
            for aid in agent_id
        }
        self.particles = {aid: [] for aid in agent_id}
    def update_belief(self, agent_id, observed_cells):
        """
        Update belief state based on observed cells.
        observed_cells: tuple of ((row,col), cell_value) pairs or (("goal_dir",), (dx,dy))
        """
        for key, value in observed_cells:
            if isinstance(key, tuple) and len(key) == 2:  # map cell (row, col)
                row, col = key
                # Belief softening: instead of setting to 0 or 1, set to 0.1 or 0.9 to maintain uncertainty
                self.belief[agent_id][row, col] = 0.1 if value == 0 else 0.9

        # Resample particles after observation to reflect updated belief
        self.particles[agent_id] = self.particle_sampling(agent_id, num_particles=100)

    def particle_sampling(self, agent_id, num_particles=100):
        """
        Sample possible world states from the belief distribution.
        Returns a list of sampled maps.
        """
        belief = self.belief[agent_id]
        rand = np.random.rand(num_particles, self.H, self.W)
        particles = (rand < belief).astype(np.uint8)
        self.particles[agent_id] = particles
        return particles

