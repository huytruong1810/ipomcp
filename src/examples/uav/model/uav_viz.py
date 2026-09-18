from examples.uav.model.uav_model import COLS, ROWS, UAVState


class UAVVisualizer:
    @staticmethod
    def print_grid(state: UAVState):
        sep = "  +" + ("---+" * COLS)
        grid = [["." for _ in range(COLS)] for _ in range(ROWS)]

        ur, uc = state.uav_pos
        tr, tc = state.target_pos

        # Place Agents
        if state.uav_pos == state.target_pos:
            grid[ur][uc] = "X"  # Crash/Catch
        else:
            grid[ur][uc] = "U"  # UAV
            grid[tr][tc] = "T"  # Target

        print("   " + "".join(f" {str(idx)}  " for idx in range(COLS)) + "\n" + sep)
        for idx, row in enumerate(grid):
            print(f"{idx} | {' | '.join(row)} |")
            print(sep)
