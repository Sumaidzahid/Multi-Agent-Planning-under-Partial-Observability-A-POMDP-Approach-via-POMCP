import numpy as np


class TreeBuilder:
    def __init__(self):

        self.root = () # root history (empty tuple)
        self.nodes = {} # history -> [parent_history, children_dict, N, V, B]
        self._init_root() # initiallize a new root

    def _init_root(self):
        self.nodes[self.root] = {
            "parent": None,      # parent history (None for root)
            "children": {},      # dict: label -> child_history
            "N": 0,              # visit count
            "V": 0.0,            # value
            "B": [],             # belief particles (for history nodes)
            "is_action": False   # False = history/observation node, True = action node
        }
  # ---------------------------------------------------------------------
    def getCreateActionNode(self, history_node, action):
        if history_node not in self.nodes:
            raise ValueError("history_node not in tree")

        children = self.nodes[history_node]["children"]

        if action not in children:
            new_hist = history_node + (action,)
            self.nodes[new_hist] = {
                "parent": history_node,
                "children": {},
                "N": 0,
                "V": 0.0,
                "B": None,
                "is_action": True
            }
            children[action] = new_hist

        return children[action]

    def getCreateObservationNode(self, action_node, observation):
        if action_node not in self.nodes:
            raise ValueError("action_node not in tree")

        children = self.nodes[action_node]["children"]

        if observation not in children:
            new_hist = action_node + (observation,)
            self.nodes[new_hist] = {
                "parent": action_node,
                "children": {},
                "N": 0,
                "V": 0.0,
                "B": [],
                "is_action": False
            }
            children[observation] = new_hist

        return children[observation]

    def make_root(self, new_root):
        if new_root not in self.nodes:
            raise ValueError("new_root not found in tree")

        # collect subtree
        stack = [new_root]
        subtree = {}
        while stack:
            nid = stack.pop()
            subtree[nid] = self.nodes[nid]
            stack.extend(self.nodes[nid]["children"].values())

        # remap histories so new_root becomes ()
        L = len(new_root)
        remap = lambda h: h[L:]

        # build new nodes
        new_nodes = {}
        for old, data in subtree.items():
            nh = remap(old)
            parent = data["parent"]
            new_parent = None if old == new_root else remap(parent)

            B = data["B"]
            new_nodes[nh] = {
                "parent": new_parent,
                "children": {},
                "N": data["N"],
                "V": data["V"],
                "B": None if data["is_action"] else B.copy(),
                "is_action": data["is_action"]
            }

        # fix children pointers
        for old, data in subtree.items():
            nh = remap(old)
            for key, child in data["children"].items():
                new_nodes[nh]["children"][key] = remap(child)

        self.nodes = new_nodes
        self.root = ()

def UCB(N, n, V, c=1.0):
    """
    N: total visits to parent
    n: visits to this child
    V: estimated value of this child
    c: exploration constant
    """
    if n == 0:
        return float('inf')
    return V + c * np.sqrt(np.log(N) / n)
