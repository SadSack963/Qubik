"""
Author: SadSack963
Version: 0.18
Date: 29/05/2026

Requirements: Tested with Python 3.14
Packages: pip install pygame-ce numpy
Tools used: LM Studio using LLM qwen3-coder-next
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
        self.winning_line_coords = []  # ✅ Store winning coords here

        # AI State
        self.aiThinking = False

        # 3D Geometry Setup
        # Camera presets
        self.view_presets = {
            # This is the standard isometric projection used in math, CAD, and data visualization
            'math': {  # Y vertical, X ~-15°, Z ~45° depth
                'angle_y': -5 * math.pi / 12,  # -75 degrees
                'angle_x': 0.0,
                'pan_x': WIDTH // 2,
                'pan_y': HEIGHT // 2 - 50,
                'zoom_level': 1.2
            },
            'classic_isometric': {  # Symmetric isometric (X/Y/Z equal)
                'angle_y': math.pi / 4,  # +45°
                'angle_x': math.atan(math.sqrt(2) / 2),  # ≈35.264° (ideal iso)
                'pan_x': WIDTH // 2,
                'pan_y': HEIGHT // 2 - 50,
                'zoom_level': 1.0
            },
        }

        # Current view state (start with math view)
        self.current_view = 'math'
        self.angle_y = self.view_presets['math']['angle_y']
        self.angle_x = self.view_presets['math']['angle_x']
        self.pan_x = self.view_presets['math']['pan_x']
        self.pan_y = self.view_presets['math']['pan_y']
        self.zoom_level = self.view_presets['math']['zoom_level']

        # ✅ Add: axes visibility toggle
        self.show_axes = True  # start visible

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

        # Camera animation state
        self.animating_view = False  # is a view transition active?
        self.anim_start_time = None  # when did it start? (pygame time in ms)
        self.anim_duration_ms = 500  # duration: 0.5 seconds
        self.anim_target = {}  # target values to animate TO
        self.anim_initial = {}  # current values at start of anim

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

    def get_effective_camera(self):
        """Returns dict of current camera values, updating self.angle_* if animating."""
        if not self.animating_view:
            # No animation: use stored state directly
            return {
                'angle_y': self.angle_y,
                'angle_x': self.angle_x,
                'pan_x': self.pan_x,
                'pan_y': self.pan_y,
                'zoom_level': self.zoom_level
            }

        now = pygame.time.get_ticks()
        elapsed_ms = now - self.anim_start_time
        if elapsed_ms >= self.anim_duration_ms:
            # Animation complete → snap to target AND update stored state!
            self.animating_view = False

            # ←←← CRITICAL: Update actual camera variables here!
            self.angle_y = float(self.anim_target['angle_y'])
            self.angle_x = float(self.anim_target['angle_x'])
            self.pan_x = int(self.anim_target['pan_x'])
            self.pan_y = int(self.anim_target['pan_y'])
            self.zoom_level = float(self.anim_target['zoom_level'])

            return self.anim_target.copy()

        # Still animating → interpolate
        t = elapsed_ms / self.anim_duration_ms

        def lerp(a, b):
            return a + (b - a) * t

        current = {
            'angle_y': lerp(self.anim_initial['angle_y'], self.anim_target['angle_y']),
            'angle_x': lerp(self.anim_initial['angle_x'], self.anim_target['angle_x']),
            'pan_x': lerp(self.anim_initial['pan_x'], self.anim_target['pan_x']),
            'pan_y': lerp(self.anim_initial['pan_y'], self.anim_target['pan_y']),
            'zoom_level': lerp(self.anim_initial['zoom_level'], self.anim_target['zoom_level'])
        }

        # Also update stored state *in real time* so project() always sees correct values
        self.angle_y = current['angle_y']
        self.angle_x = current['angle_x']
        self.pan_x = int(current['pan_x'])
        self.pan_y = int(current['pan_y'])
        self.zoom_level = current['zoom_level']

        return current

    def project(self, point):
        """Projects a 3D world point to 2D screen using the *actual* camera state (animated or not)."""
        x, y, z = point

        # Get current effective camera (always up-to-date)
        cam = self.get_effective_camera()

        cy = math.cos(cam['angle_y'])
        sy = math.sin(cam['angle_y'])
        cx = math.cos(cam['angle_x'])
        sx = math.sin(cam['angle_x'])

        # Rotate around Y: XZ-plane
        x1 = x * cy - z * sy
        z1 = x * sy + z * cy

        # Flip Y to match Pygame
        y2 = (-y) * cx - z1 * sx
        z2 = (-y) * sx + z1 * cx

        iso_x = (x1 - z2) * math.cos(math.radians(30))
        iso_y = (x1 + z2) * math.sin(math.radians(30)) - y2

        return np.array([
            cam['pan_x'] + iso_x * cam['zoom_level'],
            cam['pan_y'] - iso_y * cam['zoom_level']
        ], dtype=float)

    def compute_depth_factor(self, pos):
        """
        Returns a value between 0.0 (closest) and ~0.7 (farthest)
        to darken distant cubes — helps depth perception.
        Based on actual 3D Z-position after rotation.
        """
        # Project the point fully (including Z in screen space)
        x, y, z = pos
        cy, sy = math.cos(self.angle_y), math.sin(self.angle_y)
        cx, sx = math.cos(self.angle_x), math.sin(self.angle_x)

        x1 = x * cy - z * sy
        z1 = x * sy + z * cy

        y2 = y * cx - z1 * sx
        # z2 is the true depth in projected space (larger Z2 = farther back)
        z2 = y * sx + z1 * cx

        # Map z2 into a reasonable depth range for darkening
        # In our setup: center ≈ 0, min ≈ -SPACING*2.5, max ≈ SPACING*2.5
        # Let's use 60% of total spread to define darkest point
        range_min = -SPACING * 2.0
        range_max = SPACING * 1.0
        t = (z2 - range_min) / (range_max - range_min)
        t = max(0.0, min(1.0, t))  # clamp

        # Darken from 0% to ~70%: darkness increases with depth (t)
        return t * 0.7

    def draw_player_piece(self, center_2d, player_num, override_color=None, depth_factor=0.0):
        char = 'X' if player_num == 1 else 'O'

        # Use provided color if given; otherwise compute it as before (fallback)
        if override_color is not None:
            base_color = override_color
        else:
            # Default behavior: only flash all for winner → we'll avoid this now by always passing
            if self.game_over and (self.winner == player_num) and self.flash_timer > 0:
                base_color = COLORS['flash_active'] if self.flash_phase else (
                    COLORS['p1_color'] if player_num == 1 else COLORS['p2_color']
                )
            else:
                base_color = COLORS['p1_color'] if player_num == 1 else COLORS['p2_color']

        # Apply depth darkening to piece too (optional but nice for consistency)
        r = int(base_color[0] * (1 - depth_factor))
        g = int(base_color[1] * (1 - depth_factor))
        b = int(base_color[2] * (1 - depth_factor))
        color = (max(0, r), max(0, g), max(0, b))

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

    def set_view(self, preset='math', animate=False):
        """
        Sets camera view (optionally with animation).

        Args:
            preset: 'math' | 'classic_isometric' | 'toggle' | custom dict
            animate: if True, interpolates over self.anim_duration_ms (0.5s)
        """
        # Belt and Braces!
        if not hasattr(self, 'current_view'):  # first-time setup
            self.current_view = 'math'  # fallback init

        # Determine target values
        if isinstance(preset, str):
            if preset == 'toggle':
                self.current_view = ('classic_isometric'
                               if self.current_view == 'math'
                               else 'math')
                target = self.view_presets[self.current_view]
            elif preset in self.view_presets:
                target = self.view_presets[preset]
            else:
                print(f"Warning: view preset '{preset}' not found. Using 'math'.")
                target = self.view_presets['math']
        else:
            # Custom dict: fill missing keys with 'math' defaults
            target = {k: self.view_presets['math'].get(k, 0) for k in
                      ['angle_y', 'angle_x', 'pan_x', 'pan_y', 'zoom_level']}
            target.update(preset)

        if animate:
            # ✅ CRITICAL: Initialize animation state FIRST (even if not animating yet)
            self.anim_start_time = pygame.time.get_ticks()
            self.anim_duration_ms = 500
            self.anim_initial = {
                'angle_y': self.angle_y,
                'angle_x': self.angle_x,
                'pan_x': self.pan_x,
                'pan_y': self.pan_y,
                'zoom_level': self.zoom_level,
            }
            self.anim_target = target.copy()
            self.animating_view = True
        else:
            # Instant snap: update state directly
            self.angle_y = float(target['angle_y'])
            self.angle_x = float(target['angle_x'])
            self.pan_x = int(target['pan_x'])
            self.pan_y = int(target['pan_y'])
            self.zoom_level = float(target['zoom_level'])

    def get_animated_view(self):
        """Returns dict of current animated camera state (lerp between initial and target)."""
        if not self.animating_view:
            return {
                'angle_y': self.angle_y,
                'angle_x': self.angle_x,
                'pan_x': self.pan_x,
                'pan_y': self.pan_y,
                'zoom_level': self.zoom_level
            }

        now = pygame.time.get_ticks()
        elapsed_ms = now - self.anim_start_time
        if elapsed_ms >= self.anim_duration_ms:
            # Animation complete → snap to target (no overshoot)
            self.animating_view = False
            return self.anim_target.copy()

        t = elapsed_ms / self.anim_duration_ms  # progress: 0.0 → 1.0

        def lerp(a, b):  # Linear Interpolation
            return a + (b - a) * t

        return {
            'angle_y': lerp(self.anim_initial['angle_y'], self.anim_target['angle_y']),
            'angle_x': lerp(self.anim_initial['angle_x'], self.anim_target['angle_x']),
            'pan_x': lerp(self.anim_initial['pan_x'], self.anim_target['pan_x']),
            'pan_y': lerp(self.anim_initial['pan_y'], self.anim_target['pan_y']),
            'zoom_level': lerp(self.anim_initial['zoom_level'], self.anim_target['zoom_level'])
        }

    def draw_cube(self, cube):
        pos = cube['pos']

        if cube['key'] in self.grid_state:
            # It's a filled cell
            state = self.grid_state[cube['key']]
            center_2d = self.project(pos)

            # ✅ Determine color based on flash and winning line
            is_winning_cube = (
                    self.game_over and
                    self.flash_timer > 0 and
                    cube['grid_idx'] in self.winning_line_coords
            )

            if is_winning_cube:
                base_color = COLORS['p1_color'] if state['player'] == 1 else COLORS['p2_color']
                color = COLORS['flash_active'] if self.flash_phase else base_color
            else:
                color = COLORS['p1_color'] if state['player'] == 1 else COLORS['p2_color']

            # ✅ Pass `color` explicitly (override default behavior)
            self.draw_player_piece(center_2d, state['player'], override_color=color,
                                   depth_factor=self.compute_depth_factor(pos))

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
                # ✅ Compute depth-based darkening factor for visual depth cueing
                depth_factor = self.compute_depth_factor(pos)

                # Base inactive color (lighter, closer)
                base_color = COLORS['dot_inactive']

                # Darken each channel by depth_factor: e.g., (140, 170, 220) → * (1 - 0.3) for mid-distance
                r = int(base_color[0] * (1 - depth_factor))
                g = int(base_color[1] * (1 - depth_factor))
                b = int(base_color[2] * (1 - depth_factor))

                # Clamp to [0, 255]
                dark_color = (max(0, r), max(0, g), max(0, b))

                base_radius = max(1, int(CUBE_SIZE * 0.20 * self.zoom_level))

                # Shadow remains slightly offset but also darkens for consistency
                shadow_r = max(0, int(COLORS['dot_shadow'][0] * (1 - depth_factor)))
                shadow_g = max(0, int(COLORS['dot_shadow'][1] * (1 - depth_factor)))
                shadow_b = max(0, int(COLORS['dot_shadow'][2] * (1 - depth_factor)))
                shadow_color = (shadow_r, shadow_g, shadow_b)

                # Draw with depth-based darkening
                pygame.draw.circle(self.screen, shadow_color,
                                   (int(center_2d[0]) + 2, int(center_2d[1]) + 2),
                                   base_radius)
                pygame.draw.circle(self.screen, dark_color, tuple(map(int, center_2d)), base_radius)

    def draw_axes(self):
        """Draws world-aligned axes: X=red, Y=green, Z=blue + labels"""
        if not self.show_axes:
            return

        # Define axis length in world units — tuned for visibility
        axis_len = SPACING * 3.5  # slightly larger than grid extent (grid spans ~SPACING*3)

        # Endpoints (world coordinates)
        origin = np.array([0.0, 0.0, 0.0])

        x_end = origin + np.array([axis_len, 0.0, 0.0])  # X axis
        y_end = origin + np.array([0.0, axis_len, 0.0])  # Y axis (up in world)
        z_end = origin + np.array([0.0, 0.0, axis_len])  # Z axis

        # Project all points to screen space
        o_2d = self.project(origin)
        x_2d = self.project(x_end)
        y_2d = self.project(y_end)
        z_2d = self.project(z_end)

        # Draw lines (thicker than guide lines)
        line_width = 3

        # Colors
        x_color = COLORS['p1_color']  # red/coral
        y_color = (80, 255, 80)  # bright green
        z_color = COLORS['p2_color']  # blue

        pygame.draw.line(self.screen, x_color, tuple(map(int, o_2d)), tuple(map(int, x_2d)), line_width)
        pygame.draw.line(self.screen, y_color, tuple(map(int, o_2d)), tuple(map(int, y_2d)), line_width)
        pygame.draw.line(self.screen, z_color, tuple(map(int, o_2d)), tuple(map(int, z_2d)), line_width)

        # Helper to draw arrowhead (same as before)
        def draw_arrowhead(center, end_point, color):
            vec = np.array(end_point) - np.array(center)
            length = math.hypot(*vec)
            if length < 5:
                return

            dx, dy = vec / length
            perp_x, perp_y = -dy, dx
            head_len = max(10, int(CUBE_SIZE * 0.4 * self.zoom_level))
            base_width = max(3, int(head_len * 0.6))

            tip = end_point
            base1 = end_point - np.array([dx * head_len + perp_x * base_width,
                                          dy * head_len + perp_y * base_width])
            base2 = end_point - np.array([dx * head_len - perp_x * base_width,
                                          dy * head_len - perp_y * base_width])

            pts = [tuple(map(int, tip)), tuple(map(int, base1)), tuple(map(int, base2))]
            pygame.draw.polygon(self.screen, color, pts)

        # Draw arrowheads
        draw_arrowhead(o_2d, x_2d, x_color)
        draw_arrowhead(o_2d, y_2d, y_color)
        draw_arrowhead(o_2d, z_2d, z_color)

        # --- ✅ Now: Add text labels near arrowheads ---
        # Label offset (pixels) — outward from axis origin, perpendicular to axis line
        label_offset = max(15, int(CUBE_SIZE * 0.35 * self.zoom_level))

        try:
            font = pygame.font.SysFont("Arial", int(24 * self.zoom_level), bold=True)
        except:
            font = pygame.font.Font(None, int(24 * self.zoom_level))

        # Helper to compute label position & darkness (if desired)
        def render_axis_label(text, tip_pos, color):
            # Direction vector from origin to tip
            vec = np.array(tip_pos) - np.array(o_2d)
            length = math.hypot(*vec)
            if length < 1:
                return None

            dx, dy = vec / length

            # Perpendicular offset for label (rotate +90°: (dx, dy) → (-dy, dx))
            perp_x, perp_y = -dy, dx
            offset_vec = np.array([perp_x * label_offset, perp_y * label_offset])

            pos = tip_pos + offset_vec

            # Optional depth darkening (same as pieces)
            z2 = 0.0
            for cube in self.cubes:
                if cube['key'] == "0,0,0":  # origin cube approx — or just compute directly:
                    x, y, z = cube['pos']
                    cy, sy = math.cos(self.angle_y), math.sin(self.angle_y)
                    cx, sx = math.cos(self.angle_x), math.sin(self.angle_x)
                    x1 = x * cy - z * sy
                    z1 = x * sy + z * cy
                    y2 = y * cx - z1 * sx
                    z2 = y * sx + z1 * cx
                    break

            # Normalize for depth (using same bounds as drawing loop)
            z_min, z_max = -SPACING * 1.8, SPACING * 2.0
            t = (z2 - z_min) / (z_max - z_min)
            depth_norm = max(0.0, min(1.0, t))
            dark_factor = depth_norm * 0.3  # max 30% darker

            r = int(color[0] * (1 - dark_factor))
            g = int(color[1] * (1 - dark_factor))
            b = int(color[2] * (1 - dark_factor))
            final_color = (max(0, r), max(0, g), max(0, b))

            # Render label
            text_surf = font.render(text, True, final_color)
            rect = text_surf.get_rect(center=(int(pos[0]), int(pos[1])))

            return text_surf, rect

        # ✅ Draw X label (near X arrowhead)
        x_label_data = render_axis_label("X", x_2d, x_color)
        if x_label_data:
            self.screen.blit(x_label_data[0], x_label_data[1])

        # ✅ Draw Y label (near Y arrowhead)
        y_label_data = render_axis_data = render_axis_label("Y", y_2d, y_color)
        if y_label_data:
            self.screen.blit(y_label_data[0], y_label_data[1])

        # ✅ Draw Z label (near Z arrowhead)
        z_label_data = render_axis_label("Z", z_2d, z_color)
        if z_label_data:
            self.screen.blit(z_label_data[0], z_label_data[1])

        # Draw origin marker (small sphere)
        o_radius = max(3, int(CUBE_SIZE * 0.15 * self.zoom_level))
        pygame.draw.circle(self.screen, (255, 255, 255), tuple(map(int, o_2d)), o_radius)

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
                # ✅ Store and return the exact winning coordinates
                self.winning_line_coords = tuple(sorted(line_points))
                return p
        return 0

    def handle_input(self):
        keys = pygame.key.get_pressed()

        # Camera Controls (cursor keys)
        if keys[pygame.K_LEFT]:
            self.angle_y -= ROTATION_SPEED
        elif keys[pygame.K_RIGHT]:
            self.angle_y += ROTATION_SPEED

        if keys[pygame.K_UP]:
            self.angle_x -= ROTATION_SPEED
        elif keys[pygame.K_DOWN]:
            self.angle_x += ROTATION_SPEED

        # ✅ Toggle axes: Ctrl+R
        modifiers = pygame.key.get_mods()
        if keys[pygame.K_r] and modifiers & pygame.KMOD_CTRL:
            if not hasattr(self, '_r_key_pressed'):
                self.show_axes = not self.show_axes
                self._r_key_pressed = True  # prevent repeat on hold
        elif not keys[pygame.K_r]:
            if hasattr(self, '_r_key_pressed'):
                del self._r_key_pressed

        # Pan (w / a / s / d keys)
        if keys[pygame.K_a]:
            self.pan_x -= PAN_SPEED
        elif keys[pygame.K_d]:
            self.pan_x += PAN_SPEED
        elif keys[pygame.K_w]:
            self.pan_y -= PAN_SPEED
        elif keys[pygame.K_s]:
            self.pan_y += PAN_SPEED

        # Zoom Controls (z / Z keys)
        if keys[pygame.K_z]:
            modifiers = pygame.key.get_mods()
            if modifiers & pygame.KMOD_SHIFT:
                self.zoom_level *= ZOOM_STEP
            else:
                self.zoom_level *= ZOOM_IN_FACTOR
        # Limit zoom levels
        self.zoom_level = max(0.5, min(self.zoom_level, 3.0))

        # Mode Switching (m key)
        if keys[pygame.K_m] and not hasattr(self, '_m_key_pressed'):
            self.toggle_mode()
            self._m_key_pressed = True
        elif not keys[pygame.K_m]:
            if hasattr(self, '_m_key_pressed'):
                del self._m_key_pressed

        if keys[pygame.K_v]:
            if not hasattr(self, '_view_toggle_pressed'):
                self.set_view('toggle', animate=True)
                self._view_toggle_pressed = True
        elif not keys[pygame.K_v]:
            if hasattr(self, '_view_toggle_pressed'):
                del self._view_toggle_pressed

        # Reset View (r key)
        # Restores defaults to start a new game (if game over)
        if keys[pygame.K_r] or keys[pygame.K_HOME]:
            if not hasattr(self, '_view_reset_pressed'):
                self.reset_game()
                self._view_reset_pressed = True
            elif not keys[pygame.K_r and not keys[pygame.K_HOME]]:  # release handler
                if hasattr(self, '_view_reset_pressed'):
                    del self._view_reset_pressed

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
        self.game_over = True
        self.reset_game()  # Reset on mode switch

    def reset_game(self):
        # ✅ Reset camera to clean view
        self.set_view(self.current_view, animate=True)   # ← animate the reset!

        if self.game_over:
            self.grid_state.clear()
            self.current_player = 1
            self.game_over = False
            self.winner = None
            self.flash_timer = 0
            self.aiThinking = False
            self.winning_line_coords = []  # ✅ Reset winning line coords

    def run(self):
        running = True
        font_win = pygame.font.SysFont("Arial", 32, bold=True)
        font_hud = pygame.font.SysFont("Arial", 20, bold=True)
        small_font = pygame.font.SysFont("Arial", 16, bold=False)

        LEFT_PANEL_WIDTH = 280
        RIGHT_PANEL_WIDTH = 250

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

            # Handle Flash Effect Timer
            if self.game_over and self.flash_timer > 0:
                self.flash_timer -= 1
                if self.flash_timer % 10 == 0:
                    self.flash_phase = not self.flash_phase

            # AI TURN LOGIC
            if not self.game_over and self.mode == 'PvAI' and self.current_player == 2:
                self.make_ai_move()

            self.handle_input()

            # --- CORE DRAWING SECTION (REPLACEMENT) ---

            self.screen.fill(COLORS['bg'])
            self.draw_guide_lines()

            # ✅ Draw axes (if enabled) BEFORE grid/dots/cubes
            self.draw_axes()

            # =============================================================
            # ✅ STEP 1: Precompute depth for all cubes once per frame
            #    - Used for correct Painter's Algorithm ordering + shading
            # =============================================================

            depth_map = {}

            for cube in self.cubes:
                x, y, z = cube['pos']

                # Project to get true Z-depth (screen space)
                cy, sy = math.cos(self.angle_y), math.sin(self.angle_y)
                cx, sx = math.cos(self.angle_x), math.sin(self.angle_x)

                x1 = x * cy - z * sy
                z1 = x * sy + z * cy

                y2 = y * cx - z1 * sx
                z2 = y * sx + z1 * cx  # True depth: larger = farther back

                # Normalize to [0, 1] using empirical bounds from your setup:
                # Range ≈ [-SPACING*2.0, SPACING*2.5], tuned to ~95% of points
                z_min, z_max = -SPACING * 1.8, SPACING * 2.0
                t = (z2 - z_min) / (z_max - z_min)
                depth_norm = max(0.0, min(1.0, t))  # clamp

                # Store full info for later use
                depth_map[cube['key']] = {
                    'z2': z2,
                    'depth_norm': depth_norm
                }

            # =============================================================
            # ✅ STEP 2: Sort cubes by screen-space Z (far → near)
            #    - Use negative z2 so far items (large z2) come first in list
            # =============================================================

            sorted_cubes = sorted(self.cubes, key=lambda c: -depth_map[c['key']]['z2'])

            # =============================================================
            # ✅ STEP 3: Draw all cubes with depth shading & updated sizes
            # =============================================================

            for cube in sorted_cubes:
                pos = cube['pos']
                center_2d = self.project(pos)
                z2 = depth_map[cube['key']]['z2']
                depth_norm = depth_map[cube['key']]['depth_norm']

                # Apply darkness factor (tweak multipliers for effect strength)
                dark_factor_dot = depth_norm * 0.6  # dots fade up to 60%
                dark_factor_piece = depth_norm * 0.4  # pieces fade up to 40% (less subtle)

                if cube['key'] in self.grid_state:
                    state = self.grid_state[cube['key']]

                    # Determine base color (flash or normal)
                    is_winning_cube = (
                            self.game_over and
                            self.flash_timer > 0 and
                            tuple(cube['grid_idx']) in self.winning_line_coords
                    )

                    if is_winning_cube:
                        base_color = COLORS['p1_color'] if state['player'] == 1 else COLORS['p2_color']
                        color = COLORS['flash_active'] if self.flash_phase else base_color
                    else:
                        color = COLORS['p1_color'] if state['player'] == 1 else COLORS['p2_color']

                    # Apply depth-based darkening to piece
                    r = int(color[0] * (1 - dark_factor_piece))
                    g = int(color[1] * (1 - dark_factor_piece))
                    b = int(color[2] * (1 - dark_factor_piece))
                    final_color = (max(0, r), max(0, g), max(0, b))

                    # ✅ Draw the piece with depth-adjusted color
                    self.draw_player_piece(center_2d, state['player'], override_color=final_color)

                else:
                    # Empty cell: draw dot with depth shading — NOW SMALLER!
                    base_dot = COLORS['dot_inactive']
                    shadow_dot = COLORS['dot_shadow']

                    r = int(base_dot[0] * (1 - dark_factor_dot))
                    g = int(base_dot[1] * (1 - dark_factor_dot))
                    b = int(base_dot[2] * (1 - dark_factor_dot))
                    shaded_dot_color = (max(0, r), max(0, g), max(0, b))

                    sr = int(shadow_dot[0] * (1 - dark_factor_dot * 0.8))  # shadow fades less
                    sg = int(shadow_dot[1] * (1 - dark_factor_dot * 0.8))
                    sb = int(shadow_dot[2] * (1 - dark_factor_dot * 0.8))
                    shaded_shadow_color = (max(0, sr), max(0, sg), max(0, sb))

                    # ✅ Smaller radius: 0.85 × original (was CUBE_SIZE * 0.20 → now * 0.17)
                    base_radius = max(1, int(CUBE_SIZE * 0.17 * self.zoom_level))

                    shadow_pos = (int(center_2d[0]) + 2, int(center_2d[1]) + 2)

                    # Draw shadow first
                    pygame.draw.circle(self.screen, shaded_shadow_color, shadow_pos, base_radius)
                    # Then main dot
                    pygame.draw.circle(self.screen, shaded_dot_color, tuple(map(int, center_2d)), base_radius)

            # =============================================================
            # ✅ STEP 4: Ghost (hover) effect — same size as placed, yellow!
            # =============================================================

            if not self.game_over and not self.aiThinking:
                if (
                        (self.mode == 'PvAI' and self.current_player == 1) or
                        (self.mode == 'PvP')
                ):
                    hover_cube = None
                    for cube in sorted_cubes:  # same order as drawing
                        pos = cube['pos']
                        center_2d = self.project(pos)
                        if np.array_equal(self.hovered_cube_pos, pos):
                            hover_cube = cube
                            break

                    if hover_cube:
                        depth_norm = depth_map[hover_cube['key']]['depth_norm']

                        # ✅ Ghost: same font size as placed pieces (base_size computed in draw_player_piece)
                        base_size = int(CUBE_SIZE * 0.85 * self.zoom_level)  # ← SAME SIZE!
                        font_size = max(32, base_size)

                        try:
                            font = pygame.font.SysFont("Arial", font_size, bold=True)
                        except:
                            font = pygame.font.Font(None, font_size)

                        # ✅ Yellow ghost (brighter for visibility)
                        yellow_ghost = (255, 230, 80)  # COLORS['ghost_color'] already defined as this
                        r = int(yellow_ghost[0] * (1 - depth_norm * 0.2))
                        g = int(yellow_ghost[1] * (1 - depth_norm * 0.2))
                        b = int(yellow_ghost[2] * (1 - depth_norm * 0.2))
                        ghost_color = (max(0, r), max(0, g), max(0, b))

                        # ✅ Slightly transparent for depth cueing
                        alpha = int(180 * (0.7 + 0.3 * (1 - depth_norm)))  # closer = brighter/less transparent

                        char = 'X' if self.current_player == 1 else 'O'
                        text_surf = font.render(char, True, ghost_color)
                        text_surf.set_alpha(alpha)  # apply depth-based transparency
                        rect = text_surf.get_rect(center=(int(center_2d[0]), int(center_2d[1])))
                        self.screen.blit(text_surf, rect)

            # =============================================================
            # ✅ STEP 5: UI & Overlays (unchanged)
            # =============================================================

            left_panel_x = 20
            left_panel_y = HEIGHT - 400
            s_left = pygame.Surface((LEFT_PANEL_WIDTH, 390))
            s_left.set_alpha(220)
            s_left.fill(COLORS['ui_panel_bg'])
            self.screen.blit(s_left, (left_panel_x, left_panel_y))

            instr_title = font_hud.render("Controls", True, (255, 255, 255))
            self.screen.blit(instr_title, (left_panel_x + 10, left_panel_y + 15))

            instructions = [
                "Arrows: Rotate View",
                "W / S: Pan Up / Down",
                "A / D: Pan Left / Right",
                "Z / Shift + Z: Zoom In / Out",
                "R / Home: Math View / New Game",
                f"V: View {'Math' if self.current_view == 'classic_isometric' else 'Classic'}",
                f"Ctrl+R: Turn Axes {'OFF' if self.show_axes else 'ON'}",
                "",
                f"Current Mode: {self.mode}",
                "M: Switch Mode",
            ]

            for i, line in enumerate(instructions):
                if "Current Mode" in line:
                    txt_instr = font_hud.render(line, True, (50, 255, 150))
                else:
                    txt_instr = small_font.render(line, True, (200, 200, 200))
                self.screen.blit(txt_instr, (left_panel_x + 15, left_panel_y + 60 + (i * 30)))

            if self.aiThinking:
                thinking_surf = font_hud.render("AI is Thinking...", True, COLORS['ai_thinking'])
                rect_think = thinking_surf.get_rect(center=(WIDTH // 2, HEIGHT - 260))
                self.screen.blit(thinking_surf, rect_think)
            else:
                if not self.game_over:
                    p_name = "Player 1 (You)" if self.current_player == 1 else \
                        ("AI Player" if self.mode == 'PvAI' else f"Player {self.current_player}")
                    p_char = "X" if self.current_player == 1 else "O"

                    turn_text = f"Current Turn: {p_name} ({p_char})"
                    txt_turn = font_hud.render(turn_text, True, (255, 255, 255))
                    txt_rect = txt_turn.get_rect(center=(WIDTH // 2, HEIGHT - 170))
                    self.screen.blit(txt_turn, txt_rect)

                    p_color = COLORS['p1_color'] if self.current_player == 1 else COLORS['p2_color']
                    turn_indicator_center = (txt_rect.right + 20, txt_rect.centery)
                    pygame.draw.circle(self.screen, p_color, turn_indicator_center, 10)

            # Right Panel (Scoreboard)
            right_panel_x = WIDTH - RIGHT_PANEL_WIDTH - 20
            right_panel_y = HEIGHT - 240
            s_right = pygame.Surface((RIGHT_PANEL_WIDTH, 230))
            s_right.set_alpha(220)
            s_right.fill(COLORS['ui_panel_bg'])
            self.screen.blit(s_right, (right_panel_x, right_panel_y))

            score_title = font_hud.render("Scoreboard", True, (255, 255, 255))
            self.screen.blit(score_title, (right_panel_x + 10, right_panel_y + 15))

            p2_label = "AI (O)" if self.mode == 'PvAI' else "Player 2 (O)"
            p1_score_text = font_hud.render(f"Player 1 (X): {self.player1_wins} Wins", True, COLORS['p1_color'])
            p2_score_text = font_hud.render(f"{p2_label}: {self.player2_wins} Wins", True, COLORS['p2_color'])
            draw_text = font_hud.render(f"Draws: {self.draws}", True, (200, 200, 200))

            self.screen.blit(p1_score_text, (right_panel_x + 15, right_panel_y + 60))
            self.screen.blit(p2_score_text, (right_panel_x + 15, right_panel_y + 95))
            self.screen.blit(draw_text, (right_panel_x + 15, right_panel_y + 130))

            # Game Over Message
            if self.game_over:
                if self.winner == "Draw":
                    msg_text = font_win.render("GAME OVER: IT'S A DRAW!", True, (255, 255, 255))
                else:
                    winner_name = "Player 1 Wins!" if self.winner == 1 else \
                        ("AI Wins!" if self.mode == 'PvAI' else "Player 2 Wins!")
                    win_color = COLORS['p1_color'] if self.winner == 1 else (80, 140, 255)
                    msg_text = font_win.render(f"{winner_name}", True, win_color)

                rect_win = msg_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 80))

                final_str = "GAME OVER: IT'S A DRAW!" if self.winner == "Draw" else \
                    "Player 1 Wins!" if self.winner == 1 else \
                        ("AI Wins!" if self.mode == 'PvAI' else "Player 2 Wins!")

                shadow_surface = font_win.render(final_str, True, (0, 0, 0))
                shadow_rect = shadow_surface.get_rect(center=(WIDTH // 2 + 4, HEIGHT // 2 - 80 + 4))
                self.screen.blit(shadow_surface, shadow_rect)
                self.screen.blit(msg_text, rect_win)

            pygame.display.flip()


if __name__ == "__main__":
    viewer = CubeViewer()
    viewer.run()
