import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_board(board):
    """
    Visualizes the SquadroBoard state with dynamic speed arrows 
    and dock zones.
    """
    clear_screen()
    print("\n" + " "*10 + "S Q U A D R O")
    print(" "*6 + "="*29 + "\n")
    
    # --- PLAYER 2 HEADER (Top Docks) ---
    
    # 1. P2 Speed & Direction Arrows
    p2_speed_str = "        "
    for i, s in enumerate(board.P2_SPEEDS):
        # Calculate effective speed: Base if Outbound (1), (4-Base) if Inbound (-1)
        eff_speed = s if board.p2_dir[i] == 1 else (4 - s)
        arrow = "v" if board.p2_dir[i] == 1 else "^"
        p2_speed_str += f" {eff_speed}{arrow} "
    print(p2_speed_str)

    # 2. P2 Start/Home Zone (Position 0)
    p2_home_str = "        "
    for i in range(5):
        # If P2 is at 0, draw piece, else empty bracket
        if board.p2_pos[i] == 0:
            sym = "▲" if board.p2_fin[i] else " ▼ "
            p2_home_str += f"[{sym.strip()}] "
        else:
            p2_home_str += "[ ] "
    print(p2_home_str)
    
    # Top Border of Grid
    print("       " + "┌───" * 5 + "┐")

    # --- MAIN GRID + PLAYER 1 (Rows) ---
    for r in range(5):
        # A. P1 Speed & Direction Label
        eff_speed = board.P1_SPEEDS[r] if board.p1_dir[r] == 1 else (4 - board.P1_SPEEDS[r])
        arrow = ">" if board.p1_dir[r] == 1 else "<"
        row_str = f" {eff_speed}{arrow} "

        # B. P1 Home Zone (Position 0)
        if board.p1_pos[r] == 0:
            sym = "◄" if board.p1_fin[r] else "►"
            row_str += f"[{sym}] "
        else:
            row_str += "[ ] "
        
        row_str += "│" # Left Grid Border

        # C. The 5x5 Grid (Positions 1-5)
        for c in range(5):
            cell_symbol = " . "
            
            # Check P1 (Horizontal)
            # Position 1 corresponds to grid index 0 (c)
            if board.p1_pos[r] == c + 1:
                cell_symbol = " ► " if board.p1_dir[r] == 1 else " ◄ "
            
            # Check P2 (Vertical) - overwrites P1 visually if colliding
            if board.p2_pos[c] == r + 1:
                cell_symbol = " ▼ " if board.p2_dir[c] == 1 else " ▲ "

            row_str += cell_symbol
            if c < 4: row_str += " " # Spacing between cols

        row_str += "│" # Right Grid Border

        # D. P1 Turnaround Zone (Position 6)
        if board.p1_pos[r] == 6:
            row_str += " [◄]"
        else:
            row_str += " [ ]"

        print(row_str)

    # Bottom Border of Grid
    print("       " + "└───" * 5 + "┘")

    # --- PLAYER 2 FOOTER (Bottom Docks) ---
    
    # P2 Turnaround Zone (Position 6)
    p2_end_str = "        "
    for i in range(5):
        if board.p2_pos[i] == 6:
            p2_end_str += "[▲] "
        else:
            p2_end_str += "[ ] "
    print(p2_end_str)

    # Status Display
    print("\n--- Game Status ---")
    p1_score = sum(board.p1_fin)
    p2_score = sum(board.p2_fin)
    print(f"P1 (Horizontal): {p1_score}/4  |  P2 (Vertical): {p2_score}/4")
    print("-" * 45)