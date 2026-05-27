"""
Author: SadSack963
Version: 0.11
Date: 27/05/2026

Requirements: Tested with Python 3.14
Packages: pip install pygame-ce numpy
Tools used: LM Studio using LLM qwen3-coder-next

This version works well for Player 1, Player 2 and AI.

Known issues:
1.  All player pieces flash when there is a winning line.
    Only the 4 pieces that make up the winning move should flash.
2.  Code could do with some tidying up (!) but it works :)
"""

import pygame
import math
import numpy as np
import copy

# -------------------------
# CONFIGURATION & CONSTANTS
# -------------------------

FPS = 60
WIDTH, HEIGHT = 1200, 950

CUBE_SIZE = 40
GAP_SIZE = 18
SPACING = CUBE_SIZE + GAP_SIZE

ROTATION_SPEED = 0.02
PAN_SPEED = 6
ZOOM_STEP = 0.95  # < 1.0 shrinks the view (zooms out)
ZOOM_IN_FACTOR = 1.1  # > 1.0 magnifies the view (zooms in)

COLORS = {
    'bg': (20, 20, 25),

    # Board Colors
    'dot_inactive': (140, 170, 220),
    'dot_shadow': (30, 35, 50),

    # Player Colors
    'p1_color': (255, 80, 80),  # X - Red/Coral
    'p2_color': (80, 140, 255),  # O - Blue

    # Flash Effect Colors
    'flash_active': (255, 255, 255),

    # Ghost Hover Effect
    'ghost_color': (255, 230, 80),

    # UI Colors
    'ui_panel_bg': (35, 35, 45),
    'text_color': (220, 220, 220),
    'grid_lines': (60, 60, 70, 50),  # RGB + Alpha

    # AI Status Colors
    'ai_thinking': (80, 200, 80)
}


class CubeViewer:
    def __init__(self):
        self._m_key_pressed = None

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("3D 4x4x4 Tic-Tac-Toe")
        self.clock = pygame.time.Clock()

        # Game State
        self.mode = 'PvP'  # 'PvP' or 'PvAI'
        self.current_player = 1
        self.grid_state = {}  # Key: "x,y,z", Value: {'player': int}

        self.player1_wins = 0
        self.player2_wins = 0
        self.draws = 0

        self.hovered_cube_pos = None
        self.game_over = False
        self.winner = None  # 1, 2, or "Draw"
        self.flash_timer = 0  # For the winning flash effect
        self.flash_phase = True  # True = White, False = Original Color

        # AI State
        self.aiThinking = False

        # 3D Geometry Setup
        self.angle_y = 0.785398
        self.angle_x = 0.61548
        self.zoom_level = 1.2
        self.pan_x, self.pan_y = WIDTH // 2, HEIGHT // 2 - 50

        offset = -(4 - 1) * SPACING / 2.0
        self.cubes = []

        for x in range(4):
            for y in range(4):
                for z in range(4):
                    pos = np.array([offset + x * SPACING,
                                    offset + y * SPACING,
                                    offset + z * SPACING], dtype=float)
                    key_str = f"{x},{y},{z}"
                    cube_data = {'pos': pos, 'key': key_str, 'grid_idx': (x, y, z)}
                    self.cubes.append(cube_data)

    # --- Core Logic & AI ---

    def get_available_moves(self):
        """Returns a list of grid indices [(x,y,z), ...] for empty spots"""
        moves = []
        for cube in self.cubes:
            if cube['key'] not in self.grid_state:
                moves.append(cube['grid_idx'])
        return moves

    def check_win_for_player(self, board_state, player):
        """
        Checks if a specific player has won.
        board_state is a dict like self.grid_state but can be modified for testing.
        Returns True/False
        """
        # We need to find which positions this player holds
        player_positions = [tuple(map(int, k.split(','))) for k, v in board_state.items() if v.get('player') == player]

        directions = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)
        ]

        for x, y, z in player_positions:
            for dx, dy, dz in directions:
                line_points = [(x, y, z)]

                # Check positive
                i = 1
                while True:
                    nx, ny, nz = x + dx * i, y + dy * i, z + dz * i
                    if 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4:
                        key = f"{nx},{ny},{nz}"
                        if board_state.get(key, {}).get('player') == player:
                            line_points.append((nx, ny, nz))
                            i += 1
                        else:
                            break
                    else:
                        break

                # Check negative
                i = 1
                while True:
                    nx, ny, nz = x - dx * i, y - dy * i, z - dz * i
                    if 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4:
                        key = f"{nx},{ny},{nz}"
                        if board_state.get(key, {}).get('player') == player:
                            line_points.append((nx, ny, nz))
                            i += 1
                        else:
                            break
                    else:
                        break

                if len(line_points) >= 4:
                    return True

        return False

    def check_draw_condition(self):
        return len(self.grid_state) == 64

    # --- AI LOGIC STARTS HERE ---

    MAX_DEPTH = 2  # Depth 2 is fast and competent. Depth 3 is smarter but slower.

    def minimax(self, board, depth, alpha, beta, is_maximizing):
        """
        Minimax with Alpha-Beta Pruning
        'board' is a copy of self.grid_state
        AI is Player 2 (O)
        Human is Player 1 (X)
        """

        # Terminal States: Win or Draw
        if self.check_win_for_player(board, 2):
            return 10000 + depth  # Prefer faster wins
        elif self.check_win_for_player(board, 1):
            return -10000 - depth  # Prefer slower losses
        elif len(board) == 64:  # Full board
            return 0

        if depth >= self.MAX_DEPTH:
            return self.evaluate_board(board)

        moves = [(int(k.split(',')[0]), int(k.split(',')[1]), int(k.split(',')[2])) for k in board.keys()]
        # If it's a new board (empty), prioritize center or corners to improve AI play
        if not moves:
            return -500

        if is_maximizing:  # AI Turn (Maximize Score)
            max_eval = float('-inf')
            best_move = None
            for move in self.get_available_moves_for_board(board):
                new_board = copy.deepcopy(board)
                new_board[f"{move[0]},{move[1]},{move[2]}"] = {'player': 2}

                eval_score = self.minimax(new_board, depth + 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break

            return max_eval if depth != 0 else (max_eval, best_move)

        else:  # Human Turn (Minimize Score)
            min_eval = float('inf')
            for move in self.get_available_moves_for_board(board):
                new_board = copy.deepcopy(board)
                new_board[f"{move[0]},{move[1]},{move[2]}"] = {'player': 1}

                eval_score = self.minimax(new_board, depth + 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def get_available_moves_for_board(self, board_state):
        """Helper for AI to find moves in a specific board state copy"""
        moves = []
        for cube in self.cubes:
            key = cube['key']
            if key not in board_state:
                moves.append(cube['grid_idx'])
        return moves

    def evaluate_board(self, board):
        """
        Heuristic scoring function.
        AI = 2 (Maximizing), Human = 1 (Minimizing)
        """
        score = 0

        # Count lines for each player
        # A line is defined by its 4 coordinates. There are 76 unique lines in a 4x4x4 grid.
        # We iterate through all cubes and directions to find them without duplication,
        # but for performance in heuristic, we can just scan all positions.

        # To avoid complex list generation, let's use the logic:
        # For every cell, look in every direction. If a line of 4 is formed by player P,
        # add points.

        # Let's do a simplified check: Count open lines for each player
        p2_score = 0
        p1_score = 0

        directions = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)
        ]

        # We need to be careful not to count the same line multiple times.
        # But for a heuristic score, slight over-counting is acceptable if it's consistent.
        # Better approach: Iterate through all 76 lines explicitly or generate them.

        # Let's generate unique lines
        lines_found = set()

        for x in range(4):
            for y in range(4):
                for z in range(4):
                    for dx, dy, dz in directions:
                        line_coords = []
                        valid_line = True
                        for i in range(4):
                            nx, ny, nz = x + dx * i, y + dy * i, z + dz * i
                            if 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4:
                                line_coords.append((nx, ny, nz))
                            else:
                                valid_line = False
                                break

                        if valid_line:
                            # Sort coords to create a unique hashable key for the set
                            sorted_coords = tuple(sorted(line_coords))
                            if sorted_coords not in lines_found:
                                lines_found.add(sorted_coords)

                                # Score this line
                                p2_count = sum(1 for c in sorted_coords if
                                               board.get(f"{c[0]},{c[1]},{c[2]}", {}).get('player') == 2)
                                p1_count = sum(1 for c in sorted_coords if
                                               board.get(f"{c[0]},{c[1]},{c[2]}", {}).get('player') == 1)
                                empty_count = 4 - p2_count - p1_count

                                # Heuristic Weights
                                if p2_count == 4:
                                    p2_score += 10000
                                elif p2_count == 3 and empty_count == 1:
                                    p2_score += 100
                                elif p2_count == 2 and empty_count == 2:
                                    p2_score += 10

                                if p1_count == 4:
                                    p1_score += 10000
                                elif p1_count == 3 and empty_count == 1:
                                    p1_score += 90  # High priority to block
                                elif p1_count == 2 and empty_count == 2:
                                    p1_score += 5

        return p2_score - p1_score

    def make_ai_move(self):
        """Called when it's AI's turn"""
        self.aiThinking = True

        # Run Minimax
        best_score, best_move = self.minimax(
            copy.deepcopy(self.grid_state),
            depth=0,
            alpha=float('-inf'),
            beta=float('inf'),
            is_maximizing=True
        )

        if best_move:
            x, y, z = best_move
            key = f"{x},{y},{z}"

            self.grid_state[key] = {'player': 2}

            # Check Win for AI (Player 2)
            winning_player = self.check_win_condition(x, y, z)

            if winning_player == 2:
                self.game_over = True
                self.winner = 2
                self.flash_timer = 180
                self.player2_wins += 1
            elif self.check_draw_condition():
                self.game_over = True
                self.winner = "Draw"
                self.draws += 1
            else:
                self.current_player = 1

        self.aiThinking = False

    # --- Drawing & Input ---

    def project(self, point):
        x, y, z = point
        cy, sy = math.cos(self.angle_y), math.sin(self.angle_y)
        cx, sx = math.cos(self.angle_x), math.sin(self.angle_x)

        x1 = x * cy - z * sy
        z1 = x * sy + z * cy

        y2 = y * cx - z1 * sx
        z2 = y * sx + z1 * cx

        iso_x = (x1 - z2) * math.cos(math.radians(30))
        iso_y = (x1 + z2) * math.sin(math.radians(30)) - y2

        return np.array([self.pan_x + iso_x * self.zoom_level,
                         self.pan_y - iso_y * self.zoom_level], dtype=float)

    def draw_player_piece(self, center_2d, player_num):
        char = 'X' if player_num == 1 else 'O'

        # Check for flash effect
        if self.game_over and (self.winner == player_num) and self.flash_timer > 0:
            color = COLORS['flash_active'] if self.flash_phase else (
                COLORS['p1_color'] if player_num == 1 else COLORS['p2_color'])
        else:
            color = COLORS['p1_color'] if player_num == 1 else COLORS['p2_color']

        base_size = int(CUBE_SIZE * 0.85 * self.zoom_level)
        font_size = max(32, base_size)

        try:
            font = pygame.font.SysFont("Arial", font_size, bold=True)
        except:
            font = pygame.font.Font(None, font_size)

        text_surf = font.render(char, True, color)

        rect = text_surf.get_rect(center=(int(center_2d[0]), int(center_2d[1])))
        self.screen.blit(text_surf, rect)

    def draw_ghost(self, center_2d):
        if self.game_over:
            return

        char = 'X' if self.current_player == 1 else 'O'
        color = COLORS['ghost_color']

        base_size = int(CUBE_SIZE * 0.7 * self.zoom_level)
        font_size = max(25, base_size)

        try:
            font = pygame.font.SysFont("Arial", font_size, bold=True)
        except:
            font = pygame.font.Font(None, font_size)

        text_surf = font.render(char, True, color)
        text_surf.set_alpha(180)

        rect = text_surf.get_rect(center=(int(center_2d[0]), int(center_2d[1])))
        self.screen.blit(text_surf, rect)

    def draw_cube(self, cube):
        pos = cube['pos']

        if cube['key'] in self.grid_state:
            # It's a filled cell
            state = self.grid_state[cube['key']]
            center_2d = self.project(pos)

            # Check for flash effect first
            if self.game_over and (self.winner == state['player']) and self.flash_timer > 0:
                color = COLORS['flash_active'] if self.flash_phase else (
                    COLORS['p1_color'] if state['player'] == 1 else COLORS['p2_color']
                )
            else:
                color = COLORS['p1_color'] if state['player'] == 1 else COLORS['p2_color']

            self.draw_player_piece(center_2d, state['player'])

        else:
            # It's an empty cell
            is_hovered = False

            # FIX 1: Allow ghost for BOTH players if it's their turn and not game over
            if not self.game_over and not self.aiThinking:
                # If PvP, both can hover. If PvAI, only Player 1 (Human) hovers.
                allow_hover = True
                if self.mode == 'PvAI' and self.current_player != 1:
                    allow_hover = False

                if allow_hover and self.hovered_cube_pos is not None and np.array_equal(self.hovered_cube_pos, pos):
                    is_hovered = True

            center_2d = self.project(pos)

            if is_hovered:
                # Draw Ghost Piece (The character the player would place)
                char = 'X' if self.current_player == 1 else 'O'

                base_size = int(CUBE_SIZE * 0.7 * self.zoom_level)
                font_size = max(25, base_size)

                try:
                    font = pygame.font.SysFont("Arial", font_size, bold=True)
                except:
                    font = pygame.font.Font(None, font_size)

                # Get the color based on whose turn it is
                ghost_color = COLORS['p1_color'] if self.current_player == 1 else COLORS['p2_color']

                text_surf = font.render(char, True, ghost_color)
                text_surf.set_alpha(180)  # Make transparent

                rect = text_surf.get_rect(center=(int(center_2d[0]), int(center_2d[1])))
                self.screen.blit(text_surf, rect)

            else:
                # Draw Inactive Dot (Grey/White center)
                base_radius = max(1, int(CUBE_SIZE * 0.20 * self.zoom_level))

                shadow_pos = (int(center_2d[0] + 2), int(center_2d[1] + 2))
                pygame.draw.circle(self.screen, COLORS['dot_shadow'], shadow_pos, base_radius)
                pygame.draw.circle(self.screen, COLORS['dot_inactive'], tuple(map(int, center_2d)), base_radius)

    def draw_guide_lines(self):
        line_color = COLORS['grid_lines']
        for cube in self.cubes:
            pos = cube['pos']
            x, y, z = int(pos[0] / SPACING + 1.5), int(pos[1] / SPACING + 1.5), int(pos[2] / SPACING + 1.5)

            if x < 3:
                next_pos = pos + np.array([SPACING, 0, 0])
                p1_2d, p2_2d = self.project(pos), self.project(next_pos)
                pygame.draw.line(self.screen, line_color, tuple(map(int, p1_2d)), tuple(map(int, p2_2d)), 1)

            if y < 3:
                next_pos = pos + np.array([0, SPACING, 0])
                p1_2d, p2_2d = self.project(pos), self.project(next_pos)
                pygame.draw.line(self.screen, line_color, tuple(map(int, p1_2d)), tuple(map(int, p2_2d)), 1)

            if z < 3:
                next_pos = pos + np.array([0, 0, SPACING])
                p1_2d, p2_2d = self.project(pos), self.project(next_pos)
                pygame.draw.line(self.screen, line_color, tuple(map(int, p1_2d)), tuple(map(int, p2_2d)), 1)

    def check_win_condition(self, x, y, z):
        p = self.grid_state[f"{x},{y},{z}"]["player"]
        directions = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)
        ]
        for dx, dy, dz in directions:
            line_points = [(x, y, z)]
            i = 1
            while True:
                nx, ny, nz = x + dx * i, y + dy * i, z + dz * i
                if 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4:
                    key = f"{nx},{ny},{nz}"
                    if self.grid_state.get(key, {}).get('player') == p:
                        line_points.append((nx, ny, nz))
                        i += 1
                    else:
                        break
                else:
                    break

            i = 1
            while True:
                nx, ny, nz = x - dx * i, y - dy * i, z - dz * i
                if 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4:
                    key = f"{nx},{ny},{nz}"
                    if self.grid_state.get(key, {}).get('player') == p:
                        line_points.append((nx, ny, nz))
                        i += 1
                    else:
                        break
                else:
                    break

            if len(line_points) >= 4:
                return p
        return 0

    # def handle_input(self):
    #     keys = pygame.key.get_pressed()
    #
    #     # Mode Switching (M key)
    #     if keys[pygame.K_m] and not hasattr(self, '_m_key_pressed'):
    #         self.toggle_mode()
    #         self._m_key_pressed = True
    #     elif not keys[pygame.K_m]:
    #         if hasattr(self, '_m_key_pressed'):
    #             del self._m_key_pressed
    #
    #     # Camera Controls
    #     if keys[pygame.K_LEFT]:
    #         self.angle_y -= ROTATION_SPEED
    #     elif keys[pygame.K_RIGHT]:
    #         self.angle_y += ROTATION_SPEED
    #
    #     if keys[pygame.K_UP]:
    #         self.angle_x -= ROTATION_SPEED
    #     elif keys[pygame.K_DOWN]:
    #         self.angle_x += ROTATION_SPEED
    #
    #     if keys[pygame.K_a]:
    #         self.pan_x -= PAN_SPEED
    #     elif keys[pygame.K_d]:
    #         self.pan_x += PAN_SPEED
    #     elif keys[pygame.K_w]:
    #         self.pan_y -= PAN_SPEED
    #     elif keys[pygame.K_s]:
    #         self.pan_y += PAN_SPEED
    #
    #     # Zoom Controls
    #     if keys[pygame.K_z]:
    #         modifiers = pygame.key.get_mods()
    #         if modifiers & pygame.KMOD_SHIFT:
    #             self.zoom_level *= ZOOM_STEP
    #         else:
    #             self.zoom_level *= ZOOM_IN_FACTOR
    #
    #     self.zoom_level = max(0.5, min(self.zoom_level, 3.0))
    #
    #     # Reset
    #     if keys[pygame.K_r] and self.game_over: self.reset_game()

    def handle_input(self):
        keys = pygame.key.get_pressed()

        # Mode Switching (M key)
        if keys[pygame.K_m] and not hasattr(self, '_m_key_pressed'):
            self.toggle_mode()
            self._m_key_pressed = True
        elif not keys[pygame.K_m]:
            if hasattr(self, '_m_key_pressed'):
                del self._m_key_pressed

        # Camera Controls (Unchanged)
        if keys[pygame.K_LEFT]:
            self.angle_y -= ROTATION_SPEED
        elif keys[pygame.K_RIGHT]:
            self.angle_y += ROTATION_SPEED

        if keys[pygame.K_UP]:
            self.angle_x -= ROTATION_SPEED
        elif keys[pygame.K_DOWN]:
            self.angle_x += ROTATION_SPEED

        if keys[pygame.K_a]:
            self.pan_x -= PAN_SPEED
        elif keys[pygame.K_d]:
            self.pan_x += PAN_SPEED
        elif keys[pygame.K_w]:
            self.pan_y -= PAN_SPEED
        elif keys[pygame.K_s]:
            self.pan_y += PAN_SPEED

        # Zoom Controls (Unchanged)
        if keys[pygame.K_z]:
            modifiers = pygame.key.get_mods()
            if modifiers & pygame.KMOD_SHIFT:
                self.zoom_level *= ZOOM_STEP
            else:
                self.zoom_level *= ZOOM_IN_FACTOR

        self.zoom_level = max(0.5, min(self.zoom_level, 3.0))

        # Reset (Unchanged)
        if keys[pygame.K_r] and self.game_over:
            self.reset_game()

    def handle_mouse_click(self):
        """A dedicated method to handle mouse clicks for any human player"""
        if self.game_over or self.aiThinking:
            return

        # If the current player is the AI, do not process mouse clicks
        if self.current_player == 2 and self.mode == 'PvAI':
            return

        # Get mouse position from event (needs to be passed from main loop)
        # But since we want this generic, let's assume you call it from within the event loop

        best_dist_sq = float('inf')
        target_cube = None
        mouse_pos = pygame.mouse.get_pos()

        for cube in self.cubes:
            center_screen = self.project(cube['pos'])
            if center_screen is None:
                continue

            dx = mouse_pos[0] - center_screen[0]
            dy = mouse_pos[1] - center_screen[1]
            dist_sq = dx * dx + dy * dy
            # Increase this multiplier to make the clickable area larger/smaller
            if dist_sq < (CUBE_SIZE * 2.5) ** 2 and dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                target_cube = cube

        if target_cube is not None:
            key = target_cube['key']
            if key not in self.grid_state:
                # Valid move found! Apply it.
                x, y, z = target_cube['grid_idx']
                self.apply_move(x, y, z)

    def apply_move(self, x, y, z):
        """Centralized logic to place a piece and check game state"""
        key = f"{x},{y},{z}"
        player_num = 1 if self.current_player == 1 else 2

        self.grid_state[key] = {'player': player_num}

        # Check Win
        winning_player = self.check_win_condition(x, y, z)
        if winning_player:
            self.game_over = True
            self.winner = winning_player
            self.flash_timer = 180

            # Update score based on who won
            if self.mode == 'PvAI':
                if winning_player == 1:
                    self.player1_wins += 1
                else:
                    self.player2_wins += 1
            else:
                # PvP Scoreboard updates
                if winning_player == 1:
                    self.player1_wins += 1
                else:
                    self.player2_wins += 1

        elif len(self.grid_state) == 64:
            self.game_over = True
            self.winner = "Draw"
            self.draws += 1

        # Switch turns ONLY if the game is not over
        if not self.game_over:
            self.current_player = 2 if self.current_player == 1 else 1

    def toggle_mode(self):
        if self.mode == 'PvP':
            self.mode = 'PvAI'
        else:
            self.mode = 'PvP'
        self.reset_game()  # Reset on mode switch

    def reset_game(self):
        self.grid_state.clear()
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.flash_timer = 0
        self.aiThinking = False

    def run(self):
        running = True
        font_hud = pygame.font.SysFont("Arial", 20, bold=True)
        small_font = pygame.font.SysFont("Arial", 16, bold=False)

        LEFT_PANEL_WIDTH = 300
        RIGHT_PANEL_WIDTH = 300

        # Initialize hover outside the loop
        self.hovered_cube_pos = None

        while running:
            self.clock.tick(FPS)
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_mouse_click()

                    # if not self.game_over and not self.aiThinking and self.current_player == 1:
                    #     best_dist_sq = float('inf')
                    #     target_cube = None
                    #
                    #     for cube in self.cubes:
                    #         center_screen = self.project(cube['pos'])
                    #         if center_screen is None: continue
                    #         dx = mouse_pos[0] - center_screen[0]
                    #         dy = mouse_pos[1] - center_screen[1]
                    #         dist_sq = dx * dx + dy * dy
                    #         if dist_sq < (CUBE_SIZE * 2.5) ** 2 and dist_sq < best_dist_sq:
                    #             best_dist_sq = dist_sq
                    #             target_cube = cube
                    #
                    #     if target_cube is not None:
                    #         key = target_cube['key']
                    #         if key not in self.grid_state:
                    #             self.grid_state[key] = {'player': self.current_player}
                    #             x_idx, y_idx, z_idx = target_cube['grid_idx']
                    #
                    #             winning_player = self.check_win_condition(x_idx, y_idx, z_idx)
                    #             if winning_player == 1:
                    #                 self.game_over = True;
                    #                 self.winner = 1
                    #                 self.flash_timer = 180
                    #                 self.player1_wins += 1
                    #             elif self.check_draw_condition():
                    #                 self.game_over = True;
                    #                 self.winner = "Draw"
                    #                 self.draws += 1
                    #             else:
                    #                 self.current_player = 2  # Switch to AI

                elif event.type == pygame.MOUSEMOTION:
                    # Hover effect logic
                    mouse_pos = event.pos
                    best_dist_sq = float('inf')
                    target_cube = None
                    for cube in self.cubes:
                        center_screen = self.project(cube['pos'])
                        if center_screen is None:
                            continue
                        dx = mouse_pos[0] - center_screen[0]
                        dy = mouse_pos[1] - center_screen[1]
                        dist_sq = dx * dx + dy * dy
                        if dist_sq < (CUBE_SIZE * 2.5) ** 2 and dist_sq < best_dist_sq:
                            best_dist_sq = dist_sq
                            target_cube = cube
                    self.hovered_cube_pos = target_cube['pos'] if target_cube else None

                    # if not self.aiThinking:
                    #     best_dist_sq = float('inf')
                    #     target_cube = None
                    #     for cube in self.cubes:
                    #         center_screen = self.project(cube['pos'])
                    #         if center_screen is None: continue
                    #         dx = mouse_pos[0] - center_screen[0]
                    #         dy = mouse_pos[1] - center_screen[1]
                    #         dist_sq = dx * dx + dy * dy
                    #         if dist_sq < (CUBE_SIZE * 2.5) ** 2 and dist_sq < best_dist_sq:
                    #             best_dist_sq = dist_sq
                    #             target_cube = cube
                    #     self.hovered_cube_pos = target_cube['pos'] if target_cube else None

            # Handle Flash Effect Timer
            if self.game_over and self.flash_timer > 0:
                self.flash_timer -= 1
                if self.flash_timer % 10 == 0:
                    self.flash_phase = not self.flash_phase

            # AI TURN LOGIC
            if not self.game_over and self.mode == 'PvAI' and self.current_player == 2:
                self.make_ai_move()

            self.handle_input()

            self.screen.fill(COLORS['bg'])
            self.draw_guide_lines()

            # Sort cubes for depth painting (Painter's Algorithm)
            sorted_cubes = sorted(self.cubes, key=lambda c: c['pos'][0] + c['pos'][1] + c['pos'][2], reverse=True)
            for cube in sorted_cubes:
                self.draw_cube(cube)

            # --- UI DRAWING ---

            left_panel_x = 20
            left_panel_y = HEIGHT - 320
            s_left = pygame.Surface((LEFT_PANEL_WIDTH, 310))
            s_left.set_alpha(220)
            s_left.fill(COLORS['ui_panel_bg'])
            self.screen.blit(s_left, (left_panel_x, left_panel_y))

            instr_title = font_hud.render("Controls", True, (255, 255, 255))
            self.screen.blit(instr_title, (left_panel_x + 10, left_panel_y + 15))

            instructions = [
                "Arrows: Rotate View",
                "W / S: Pan Up / Down", "A / D: Pan Left / Right",
                "Z: Zoom In", "Shift + Z: Zoom Out",
                "",
                f"Current Mode: {self.mode}",
                "Press 'M' to Switch",
            ]

            for i, line in enumerate(instructions):
                if "Mode" in line:
                    txt_instr = font_hud.render(line, True, (50, 255, 150))
                else:
                    txt_instr = small_font.render(line, True, (200, 200, 200))
                self.screen.blit(txt_instr, (left_panel_x + 15, left_panel_y + 60 + (i * 30)))

            if self.aiThinking:
                thinking_surf = font_hud.render("AI is Thinking...", True, COLORS['ai_thinking'])
                rect_think = thinking_surf.get_rect(center=(WIDTH // 2, HEIGHT - 260))
                self.screen.blit(thinking_surf, rect_think)
            else:
                # Current Turn Text
                if not self.game_over:
                    # p_name = "Player 1 (You)" if self.current_player == 1 else "AI Player"
                    if self.mode == 'PvAI':
                        p_name = "Player 1 (You)" if self.current_player == 1 else "AI Player"
                        p_color = COLORS['p1_color'] if self.current_player == 1 else COLORS['p2_color']
                    else:
                        p_name = f"Player {self.current_player}"
                        p_color = COLORS['p1_color'] if self.current_player == 1 else COLORS['p2_color']
                    p_char = "X" if self.current_player == 1 else "O"
                    # p_color = COLORS['p1_color'] if self.current_player == 1 else (200, 200, 50)

                    # turn_text = f"Current Turn: {p_name} ({p_char})"
                    turn_text = f"Current Turn: {p_name} ({'X' if self.current_player == 1 else 'O'})"
                    txt_turn = font_hud.render(turn_text, True, (255, 255, 255))
                    txt_rect = txt_turn.get_rect(center=(WIDTH // 2, HEIGHT - 260))
                    self.screen.blit(txt_turn, txt_rect)

                    turn_indicator_center = (txt_rect.right + 20, txt_rect.centery)
                    pygame.draw.circle(self.screen, p_color, turn_indicator_center, 10)

            # Right Panel (Scoreboard) remains unchanged
            right_panel_x = WIDTH - RIGHT_PANEL_WIDTH - 20
            right_panel_y = HEIGHT - 240
            s_right = pygame.Surface((RIGHT_PANEL_WIDTH, 230))
            s_right.set_alpha(220)
            s_right.fill(COLORS['ui_panel_bg'])
            self.screen.blit(s_right, (right_panel_x, right_panel_y))

            score_title = font_hud.render("Scoreboard", True, (255, 255, 255))
            self.screen.blit(score_title, (right_panel_x + 10, right_panel_y + 15))

            p1_score_text = font_hud.render(f"Player 1 (X): {self.player1_wins} Wins", True, COLORS['p1_color'])
            # p2_score_text = font_hud.render(f"AI (O): {self.player2_wins} Wins", True, COLORS['p2_color'])
            p2_label = "AI (O)" if self.mode == 'PvAI' else "Player 2 (O)"
            p2_score_text = font_hud.render(f"{p2_label}: {self.player2_wins} Wins", True, COLORS['p2_color'])
            draw_text = font_hud.render(f"Draws: {self.draws}", True, (200, 200, 200))

            self.screen.blit(p1_score_text, (right_panel_x + 15, right_panel_y + 60))
            self.screen.blit(p2_score_text, (right_panel_x + 15, right_panel_y + 95))
            self.screen.blit(draw_text, (right_panel_x + 15, right_panel_y + 130))

            # Game Over Message
            if self.game_over:
                if self.winner == "Draw":
                    msg_text = font_hud.render("GAME OVER: IT'S A DRAW!", True, (255, 255, 255))
                else:
                    winner_name = "You!" if self.winner == 1 else "AI Wins!"
                    win_color = COLORS['p1_color'] if self.winner == 1 else (80, 140, 255)
                    msg_text = font_hud.render(f"{winner_name}", True, win_color)

                rect_win = msg_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 80))
                final_str = "GAME OVER: IT'S A DRAW!" if self.winner == "Draw" else f"You Win!" if self.winner == 1 else "AI Wins!"
                shadow_surface = font_hud.render(final_str, True, (0, 0, 0))
                shadow_rect = shadow_surface.get_rect(center=(WIDTH // 2 + 4, HEIGHT // 2 - 80 + 4))
                self.screen.blit(shadow_surface, shadow_rect)
                self.screen.blit(msg_text, rect_win)

            pygame.display.flip()


if __name__ == "__main__":
    viewer = CubeViewer()
    viewer.run()
