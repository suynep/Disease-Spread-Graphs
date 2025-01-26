import pygame
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Constants
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 20
CELL_SIZE = WIDTH // GRID_SIZE
FPS = 2

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH + 500, HEIGHT))  # Extra space for graph
pygame.display.set_caption("Game of Life with Graph")
clock = pygame.time.Clock()

def load_grid_from_csv(filename):
    try:
        df = pd.read_csv(filename, header=None)
        return df.values.astype(int)
    except FileNotFoundError:
        return np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.8, 0.2])

def save_grid_to_csv(grid, filename):
    df = pd.DataFrame(grid)
    df.to_csv(filename, index=False, header=False)

def update_grid(grid):
    new_grid = grid.copy()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            live_neighbors = np.sum(grid[max(0, i-1):min(i+2, grid.shape[0]), max(0, j-1):min(j+2, grid.shape[1])]) - grid[i, j]
            if grid[i, j] == 1 and (live_neighbors < 2 or live_neighbors > 3):
                new_grid[i, j] = 0
            elif grid[i, j] == 0 and live_neighbors == 3:
                new_grid[i, j] = 1
    return new_grid

# Graph setup
fig, ax = plt.subplots(figsize=(6, 6))  # Increased graph size
canvas = FigureCanvas(fig)
G = nx.Graph()

def build_graph(grid):
    G.clear()
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            node = (i, j)
            G.add_node(node, color='green' if grid[i, j] == 1 else 'red')
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                        G.add_edge(node, (ni, nj))
            if i + 2 < GRID_SIZE:
                G.add_edge(node, (i + 2, j))
            if j + 2 < GRID_SIZE:
                G.add_edge(node, (i, j + 2))

def draw_graph():
    ax.clear()
    colors = [G.nodes[n]['color'] for n in G.nodes()]
    nx.draw(G, pos={node: (node[1], -node[0]) for node in G.nodes()}, ax=ax, node_size=150, node_color=colors, edge_color='gray', width=1)  # Increased node size and edge width
    canvas.draw()
    buf = canvas.buffer_rgba()
    return pygame.image.frombuffer(buf.tobytes(), canvas.get_width_height(), "RGBA")

# Load initial grid state from CSV
# grid = load_grid_from_csv("two_gliders.csv")
grid = load_grid_from_csv("glider_block.csv")

running = True
while running:
    screen.fill(BLACK)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    grid = update_grid(grid)
    build_graph(grid)
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color = WHITE if grid[i, j] == 1 else BLACK
            pygame.draw.rect(screen, color, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    graph_surface = draw_graph()
    # graph_x = (WIDTH + 500) // 2 - graph_surface.get_width() // 2
    # screen.blit(graph_surface, (graph_x, 0))


    screen.blit(graph_surface, (WIDTH - 10, 0))  # Display the graph next to the grid
    
    pygame.display.flip()
    clock.tick(FPS)

# Save final grid state
save_grid_to_csv(grid, "final_state.csv")

pygame.quit()
