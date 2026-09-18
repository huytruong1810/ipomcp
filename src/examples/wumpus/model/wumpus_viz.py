from examples.wumpus.model.constants import EAST, NORTH, SOUTH, WEST
from examples.wumpus.model.wumpus_state import WumpusState


class WumpusVisualizer:
    @staticmethod
    def print_grid(state: WumpusState):
        w, h = state.grid_size

        # Initialize grid with dots
        grid = [[" . " for _ in range(w)] for _ in range(h)]

        # 1. Place Static Features
        if not state.has_gold:
            gx, gy = state.gold_location
            grid[gy][gx] = " G "

        for px, py in state.pit_locations:
            grid[py][px] = " O "  # O for Hole/Pit

        # 2. Place Wumpus
        wx, wy = state.wumpus_pose.pos()
        if state.wumpus_alive:
            w_char = WumpusVisualizer._get_dir_char(state.wumpus_pose.orientation)
            grid[wy][wx] = f"W{w_char} "
        else:
            grid[wy][wx] = "Wx "  # Dead Wumpus

        # 3. Place Human (Overlay if same cell)
        hx, hy = state.human_pose.pos()
        if state.human_alive:
            h_char = WumpusVisualizer._get_dir_char(state.human_pose.orientation)

            # Handle collision visuals
            if (hx, hy) == (wx, wy):
                grid[hy][hx] = "X!!"  # Collision!
            elif grid[hy][hx] == " O ":
                grid[hy][hx] = "H@O"  # Fell in pit
            elif grid[hy][hx] == " G ":
                grid[hy][hx] = f"H{h_char}+"  # Standing on gold
            elif "W" in grid[hy][hx]:
                grid[hy][hx] = f"H{h_char}W"  # Standing on dead wumpus
            else:
                grid[hy][hx] = f"H{h_char} "
        else:
            if grid[hy][hx] == " O ":
                grid[hy][hx] = "RIP"
            else:
                grid[hy][hx] = "Hxx"

        # 4. Print (Top row is y=height-1)
        print("\n+" + "---+" * w)
        for y in reversed(range(h)):
            row_str = "|"
            for x in range(w):
                row_str += f"{grid[y][x]}|"
            print(row_str)
            print("+" + "---+" * w)

        # Status Bar
        status = []
        if state.has_gold:
            status.append("[VICTORY: GOLD FOUND]")
        if state.has_arrow:
            status.append("[ARROW]")
        if not state.human_alive:
            status.append("!!! DIED !!!")
        if not state.wumpus_alive:
            status.append("(Wumpus Dead)")

        print("Status: " + " ".join(status))

    @staticmethod
    def _get_dir_char(orientation: int) -> str:
        if orientation == NORTH:
            return "^"
        if orientation == EAST:
            return ">"
        if orientation == SOUTH:
            return "v"
        if orientation == WEST:
            return "<"
        return "?"
