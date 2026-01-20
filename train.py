import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from game import SquadroBoard
from model import SquadroNet
from mcts import NeuralMCTS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "squadro_net.pth"

def train_self_play(iterations=500):
    print(f"Training on {DEVICE}...")
    
    # Initialize TensorBoard Writer
    # Logs will be saved to the 'runs' folder
    writer = SummaryWriter("runs/squadro_experiment") 
    
    model = SquadroNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            print("Loaded existing model.")
        except: pass
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mcts = NeuralMCTS(model, DEVICE)
    
    # Track win rates for TensorBoard
    win_history = [] 

    for i in range(iterations):
        board = SquadroBoard()
        history = [] 
        
        # Self-play game
        while board.winner is None:
            # Fast simulation for training
            move = mcts.search(board, simulations=50) 
            
            vec = board.get_state_vector()
            target_pi = [0]*5
            target_pi[move] = 1.0
            
            history.append([vec, target_pi, 0])
            board.do_move(move)
            
        final_val = 1.0 if board.winner == 1 else -1.0
        for step in history:
            step[2] = final_val
            
        # Training Step
        inputs = torch.FloatTensor(np.array([h[0] for h in history])).to(DEVICE)
        target_pis = torch.FloatTensor(np.array([h[1] for h in history])).to(DEVICE)
        target_vs = torch.FloatTensor(np.array([h[2] for h in history])).unsqueeze(1).to(DEVICE)
        
        pred_pis, pred_vs = model(inputs)
        
        loss_v = F.mse_loss(pred_vs, target_vs)
        loss_p = -torch.sum(target_pis * pred_pis) / target_pis.size(0)
        loss = loss_v + loss_p
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- LOGGING TO TENSORBOARD ---
        writer.add_scalar("Loss/Total", loss.item(), i)
        writer.add_scalar("Loss/Value_MSE", loss_v.item(), i)
        writer.add_scalar("Loss/Policy_CrossEntropy", loss_p.item(), i)
        
        # Log Win Rate (moving average of last 20 games)
        # 1.0 = P1 win, 0.0 = P2 win
        win_history.append(1 if board.winner == 1 else 0)
        if len(win_history) > 20: win_history.pop(0)
        avg_win = sum(win_history) / len(win_history)
        writer.add_scalar("Training/P1_Win_Rate", avg_win, i)
        # -----------------------------
        
        if (i+1) % 10 == 0:
            print(f"Iter {i+1}/{iterations} | Loss: {loss.item():.4f} | Winner: P{board.winner}")
            torch.save(model.state_dict(), MODEL_PATH)

    writer.close() # Close the writer when done

if __name__ == "__main__":
    try:
        iters = int(input("How many games to train? (e.g., 500): "))
        train_self_play(iters)
    except ValueError:
        print("Invalid number.")