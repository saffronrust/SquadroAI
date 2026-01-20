import math
import torch
import numpy as np

class MCTSNode:
    def __init__(self, state, parent=None, move=None, prob=0):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = {} 
        self.visits = 0
        self.value_sum = 0
        self.prob = prob 
        self.untried_moves = state.get_legal_moves()

    def is_leaf(self):
        return len(self.children) == 0

class NeuralMCTS:
    def __init__(self, model, device, c_puct=1.0):
        self.model = model
        self.device = device
        self.c_puct = c_puct

    def search(self, root_state, simulations=400):
        root = MCTSNode(root_state.clone())
        
        # Evaluate Root
        with torch.no_grad():
            state_tensor = torch.FloatTensor(root_state.get_state_vector()).to(self.device).unsqueeze(0)
            pi, v = self.model(state_tensor)
            probs = torch.exp(pi).cpu().numpy()[0]
        
        legal = root.untried_moves
        legal_probs = [probs[m] for m in legal]
        s_p = sum(legal_probs)
        if s_p > 0:
            legal_probs = [x/s_p for x in legal_probs]
        else:
            legal_probs = [1/len(legal)]*len(legal)

        for i, move in enumerate(legal):
            root.children[move] = MCTSNode(root.state.clone(), parent=root, move=move, prob=legal_probs[i])
            root.children[move].state.do_move(move)

        for _ in range(simulations):
            node = root
            
            # Selection
            while not node.is_leaf() and node.state.winner is None:
                node = self.select_child(node)

            # Evaluation & Expansion
            if node.state.winner:
                if node.state.winner == 1: value = 1.0
                else: value = -1.0
            else:
                state_vec = node.state.get_state_vector()
                state_tensor = torch.FloatTensor(state_vec).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    pi, v = self.model(state_tensor)
                    value = v.item()
                    probs = torch.exp(pi).cpu().numpy()[0]

                legal = node.state.get_legal_moves()
                if legal:
                    l_probs = [probs[m] for m in legal]
                    s = sum(l_probs)
                    if s > 0: l_probs = [x/s for x in l_probs]
                    else: l_probs = [1/len(legal)]*len(legal)
                    
                    for i, m in enumerate(legal):
                        new_state = node.state.clone()
                        new_state.do_move(m)
                        node.children[m] = MCTSNode(new_state, parent=node, move=m, prob=l_probs[i])
            
            # Backprop
            while node is not None:
                node.visits += 1
                node.value_sum += value 
                node = node.parent

        counts = [(m, c.visits) for m, c in root.children.items()]
        return max(counts, key=lambda x: x[1])[0]

    def select_child(self, node):
        best_score = -float('inf')
        best_child = None
        
        for move, child in node.children.items():
            if child.visits == 0:
                q_val = 0
            else:
                q_val = child.value_sum / child.visits
                if node.state.turn == 2:
                    q_val = -q_val 

            u_val = self.c_puct * child.prob * math.sqrt(node.visits) / (1 + child.visits)
            score = q_val + u_val
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child