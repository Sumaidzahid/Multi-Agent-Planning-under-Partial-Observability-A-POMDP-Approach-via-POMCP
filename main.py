
import pygame as pg
from Visualization_Map import MapVisualizer as MV
from DataLoading import DataLoader as DL
from BeliefStateManager import BeliefManager as BM
from State_Manager import StateManager as SM
from MAC import MultiAgentController as MAC

def run_episode(controller: MAC,state_mgr: SM,belief_mgr: BM,viz: MV,max_steps=50,n_simulations=200,verbose=True):
    paused = False
    total_rewards = {aid: 0.0 for aid in controller.agent_ids}

    for t in range(max_steps):

        # check termination
        if state_mgr.all_agents_at_goal():
            if verbose:
                print(f"Episode ended at step {t}: all agents at goal.")
            break

        # one decentralized POMCP step
        joint_action, observations, rewards = controller.step(n_simulations)


        # accumulate rewards
        for aid, r in rewards.items():
            total_rewards[aid] += r

        # update visualization
        new_positions = [(aid, state_mgr.agent_pos[aid]) for aid in controller.agent_ids]
        viz.update_agents(new_positions, trails=controller.trails)

        if verbose:
            print(f"Step {t}")
            print("  Actions: ", {aid: str(a) for aid, a in joint_action.items()})
            print("  Rewards:", {aid: float(r) for aid, r in rewards.items()})

        # allow pygame to process events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return total_rewards

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    paused = not paused
        while paused:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return total_rewards
                if event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                    paused = False
            viz.clock.tick(60)


    if verbose:
        print("Total rewards:", {aid: float(r) for aid, r in total_rewards.items()})
    running = True 
    print("Episode finished") 
    while running: 
        for event in pg.event.get(): 
            if event.type == pg.QUIT: 
                running = False 
                viz.clock.tick(60)
    return total_rewards


if __name__ == "__main__":
    loader = DL("MAP_KRR.xlsx")
    viz = MV(loader, cell_size=40)

    agents, goals = viz.agent_start_goal()
    viz.draw_map()
    
    # Create belief manager
    belief_mgr = BM(viz.grid, list(agents.keys()))

    # Create state manager
    state_mgr = SM(viz.grid, agents, goals, belief_mgr)

    # Create multi-agent controller
    controller = MAC(state_mgr, belief_mgr, agents.keys(),gamma=0.99, horizon=3)
    # Run episode with visualization
    run_episode(controller, state_mgr, belief_mgr, viz,max_steps=500,n_simulations=22,verbose=True)

