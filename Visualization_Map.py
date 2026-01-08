"""
Interactive map visualization for drone navigation.

This module provides a pygame-based visualization
for the grid world and drone paths. It uses the same DataLoader
as Drone_UNC_Nav.py to ensure consistent data handling.
"""
import pygame as pg
import numpy as np
from DataLoading import DataLoader 


class MapVisualizer:
    def __init__(self, Loader: DataLoader, cell_size= None):
        data = Loader.load_data()
        grid = data.to_numpy(dtype=int)
        self.grid = grid
        self.H,self.W = grid.shape
        self.clock = pg.time.Clock() 
        self.fps = 60 # default speed max 60

        pg.init()
        '''
        cell size auto scaling logic in case needed. other wise manual cell_size can be provided
        '''
        # implement auto-scaling logic
        display_width, display_height = pg.display.get_desktop_sizes()[0]

        max_width = int(display_width * 0.9)    # 90% of screen width
        max_height = int(display_height * 0.9)  # 90% of screen height

        # If no cell size provided -> auto compute
        if cell_size is None:
            cell_w = max_width // self.W
            cell_h = max_height // self.H
            self.cell_size = max(5, min(cell_w, cell_h))  # never smaller than 5px
        else:
            self.cell_size = cell_size


        self.screen_width = self.W * self.cell_size +84  # extra space for legend
        self.screen_height = self.H * self.cell_size
        screen_size = (self.screen_width, self.screen_height)

        self.screen = pg.display.set_mode(screen_size)
        self.font = pg.font.SysFont("Arial", 20)  # legent font
        pg.display.set_caption("Drone Navigation Map")  # display title
        self.bg_color = (200, 200, 200)
        self.obstacle_color = (50, 50, 50)
        self.agent_colors = [(255,0,0), (0,255,0), (0,0,255), (255,165,0), (128,0,128), (0,255,255), (255,0,255), (165,42,42)]

        self.agents = {} # agent_id -> (row,col)
        self.goals = {} # goal_id -> (row,col)
    def agent_start_goal(self):
        
        
        agents_list = np.argwhere(self.grid==2) # find list of all agent start positions
        goals_list = np.argwhere(self.grid==3) # find list of all goal positions

        # Assign agents and goals with unique IDs
        self.agents = {i+1: tuple(pos) for i, pos in enumerate(agents_list)}
        self.goals  = {i+1: tuple(pos) for i, pos in enumerate(goals_list)}

        return self.agents, self.goals

    def draw_map(self):
        ''' Draw the grid, obstacles, agents, and goals
        draw the agents as circles and goals as squares

        '''
        self.screen.fill(self.bg_color)

        # Draw obstacles 
        for r in range(self.H):
            for c in range(self.W):
                if self.grid[r, c] == 1:
                    pg.draw.rect(self.screen, self.obstacle_color,
                                     (c*self.cell_size, r*self.cell_size, self.cell_size, self.cell_size)) # pg.draw.rec(x, (x,y,w,h))
                    
        # Fill legend area with a slightly different color
        legend_x = self.W * self.cell_size # start of legend area
        legend_width = self.screen_width - legend_x # width of legend area
        pg.draw.rect(self.screen, self.obstacle_color, (legend_x, 0, legend_width, self.screen_height)) #pg.draw.rec(x, (x,y,w,h))


        # Draw agents and goals
        for i, agent_id in enumerate(self.agents.keys()):
            # grab agent and goal positions
            agent_pos = self.agents[agent_id]
            goal_pos = self.goals[agent_id]
            #get the colors for the agents and goals
            color = self.agent_colors[i% len(self.agent_colors)]
            
            # legend position
            x = self.W * self.cell_size -20  # left margin
            y = 10 + i * 25  # vertical spacing between agent  
            


            # Draw color box
            pg.draw.rect(self.screen, color, (x, y, 20, 20))
            # Draw agent ID next to the box
            text_surface = self.font.render(f"Agent {agent_id}", True, (0, 0, 0)) # .render(text, antialias, color)
            self.screen.blit(text_surface, (x + 25, y)) # .blit(surface, (x,y))

            # draw the agent as circle
            pg.draw.circle(self.screen, color,
                               (agent_pos[1]*self.cell_size + self.cell_size//2,
                                agent_pos[0]*self.cell_size + self.cell_size//2),
                               self.cell_size//3, 3)  # pygame uses (x,y) so col,row
            # draw the goal as square
            pg.draw.rect(self.screen, color,
                             (goal_pos[1]*self.cell_size + self.cell_size//4,
                              goal_pos[0]*self.cell_size + self.cell_size//4,
                              self.cell_size//2, self.cell_size//2)) # pygame uses (x,y) so col,row

        pg.display.flip() # update the full display
        self.clock.tick(self.fps)

    def update_agents(self, new_positions, trails=None):
        """
        Update agents to new positions (list of tuples)
        new_positions: list of (agent_id, (row,col)) for each agent
        """
        for agent_id, pos in new_positions:
            self.agents[agent_id] = pos # update position by agent_id before redrawing
        self.draw_map()

         # Draw trails if provided
        if trails:
            for agent_id, path in trails.items(): # path is list of (row,col)
                color = self.agent_colors[(agent_id-1) % len(self.agent_colors)] # consistent color per agent
                if len(path) > 1:
                    pg.draw.lines(self.screen, color,
                                  False,
                                  [(c*self.cell_size + self.cell_size//2,
                                    r*self.cell_size + self.cell_size//2) for r,c in path], 3) # .draw.lines(surface, color, closed, pointlist, width=0)
        pg.display.flip() # update the full display
        self.clock.tick(self.fps)
    def set_fps(self, fps):
        self.fps = max(1, min(60, fps))

    

