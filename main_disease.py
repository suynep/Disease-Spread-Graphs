import pygame
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Constants
WIDTH, HEIGHT = 600, 600
GRID_NUMBER = 30  # Number of rows/columns
CELL_SIZE = WIDTH // GRID_NUMBER  # Adjust cell size based on grid number
FPS = 10

# Colors
BLACK = (0, 0, 0)  # Dead
WHITE = (200, 200, 200)  # Susceptible
RED = (255, 0, 0)  # Infected (COVID-19)
GREEN = (0, 255, 0)  # Immune
BLUE = (0, 0, 255)  # Infected (HIV)
YELLOW = (255, 255, 0)  # Infected (Bird Flu)
DODGER_BLUE = (30, 144, 255)  # Vaccinated

# Disease Properties
DISEASES = {
    "COVID-19": {
        "infection_prob": 0.3,
        "recovery_prob": 0.1,
        "death_prob": 0.02,
        "color": RED,
    },
    "HIV": {
        "infection_prob": 0.1,
        "recovery_prob": 0.01,
        "death_prob": 0.05,
        "color": BLUE,
    },
    "Bird Flu": {
        "infection_prob": 0.4,
        "recovery_prob": 0.2,
        "death_prob": 0.1,
        "color": YELLOW,
    },
}

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH + 500, HEIGHT))  # Extra space for graph
pygame.display.set_caption("Disease Spread Simulation with Vaccination")
clock = pygame.time.Clock()

def load_grid_from_csv(filename):
    try:
        df = pd.read_csv(filename, header=None)
        return df.values.astype(int)
    except FileNotFoundError:
        return np.random.choice([0, 1], size=(GRID_NUMBER, GRID_NUMBER), p=[0.8, 0.2])

def save_grid_to_csv(grid, filename):
    df = pd.DataFrame(grid)
    df.to_csv(filename, index=False, header=False)

def update_grid(grid, disease):
    new_grid = grid.copy()
    infection_prob = DISEASES[disease]["infection_prob"]
    recovery_prob = DISEASES[disease]["recovery_prob"]
    death_prob = DISEASES[disease]["death_prob"]

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:  # Infected
                # Check if the infected individual recovers or dies
                if np.random.rand() < recovery_prob:
                    if np.random.rand() < death_prob:
                        new_grid[i, j] = 3  # Dead
                    else:
                        new_grid[i, j] = 2  # Immune
            elif grid[i, j] == 2:  # Immune
                # Check if the immune individual becomes susceptible again
                if np.random.rand() < 0.02:  # 2% chance of losing immunity
                    new_grid[i, j] = 0  # Susceptible
            elif grid[i, j] == 0:  # Susceptible
                # Check neighbors for infection
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Only horizontal and vertical neighbors
                    ni, nj = i + di, j + dj
                    if 0 <= ni < GRID_NUMBER and 0 <= nj < GRID_NUMBER and grid[ni, nj] == 1:
                        if np.random.rand() < infection_prob:
                            new_grid[i, j] = 1
                            break
            # Vaccinated individuals (state 4) remain unchanged
    return new_grid

# Graph setup
fig, ax = plt.subplots(figsize=(6, 6))  # Increased graph size
canvas = FigureCanvas(fig)
G = nx.Graph()

def build_graph(grid, disease):
    G.clear()
    disease_color = DISEASES[disease]["color"]
    for i in range(GRID_NUMBER):
        for j in range(GRID_NUMBER):
            node = (i, j)
            if grid[i, j] == 0:  # Susceptible
                G.add_node(node, color=WHITE)  # Use RGB tuple for white
            elif grid[i, j] == 1:  # Infected
                G.add_node(node, color=disease_color)  # Use RGB tuple for disease color
                # Check immediate neighbors (horizontal and vertical only)
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < GRID_NUMBER and 0 <= nj < GRID_NUMBER and grid[ni, nj] == 1:
                        G.add_edge(node, (ni, nj))  # Only add edges between infected neighbors
            elif grid[i, j] == 2:  # Immune
                G.add_node(node, color=GREEN)  # Use RGB tuple for green
            elif grid[i, j] == 3:  # Dead
                G.add_node(node, color=BLACK)  # Use RGB tuple for black
            elif grid[i, j] == 4:  # Vaccinated
                G.add_node(node, color=DODGER_BLUE)  # Use RGB tuple for Dodger Blue

def draw_graph():
    ax.clear()
    colors = [tuple(c / 255 for c in G.nodes[n]['color']) for n in G.nodes()]  # Normalize RGB to [0, 1]
    nx.draw(
        G,
        pos={node: (node[1], -node[0]) for node in G.nodes()},
        ax=ax,
        node_size=150,
        node_color=colors,
        edge_color='gray',
        width=1
    )
    canvas.draw()
    buf = canvas.buffer_rgba()
    return pygame.image.frombuffer(buf.tobytes(), canvas.get_width_height(), "RGBA")

# Load initial grid state from CSV
grid = load_grid_from_csv("initial_stateasdflkj.csv")

# Track population over time
population_data = {disease: {"susceptible": [], "infected": [], "immune": [], "dead": [], "vaccinated": []} for disease in DISEASES}

running = True
simulation_running = False  # Variable to control the simulation state
current_disease = None  # No disease selected initially

while running:
    screen.fill(BLACK)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Get the position of the mouse click
            x, y = pygame.mouse.get_pos()
            # Convert the position to grid coordinates
            i, j = y // CELL_SIZE, x // CELL_SIZE
            # Toggle the state of the cell (only allow setting to susceptible, infected, or vaccinated)
            if 0 <= i < GRID_NUMBER and 0 <= j < GRID_NUMBER:
                if grid[i, j] == 0:
                    grid[i, j] = 1  # Set to infected
                elif grid[i, j] == 1:
                    grid[i, j] = 0  # Set to susceptible
                elif grid[i, j] == 4:
                    grid[i, j] = 0  # Set to susceptible (optional)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Toggle the simulation state
                simulation_running = not simulation_running
            elif event.key == pygame.K_c:
                current_disease = "COVID-19"
            elif event.key == pygame.K_h:
                current_disease = "HIV"
            elif event.key == pygame.K_b:
                current_disease = "Bird Flu"
            elif event.key == pygame.K_v:
                # Vaccinate a cell (set to state 4)
                x, y = pygame.mouse.get_pos()
                i, j = y // CELL_SIZE, x // CELL_SIZE
                if 0 <= i < GRID_NUMBER and 0 <= j < GRID_NUMBER:
                    grid[i, j] = 4  # Set to vaccinated

    if simulation_running and current_disease:
        grid = update_grid(grid, current_disease)
        # Update population data
        population_data[current_disease]["susceptible"].append(np.sum(grid == 0))
        population_data[current_disease]["infected"].append(np.sum(grid == 1))
        population_data[current_disease]["immune"].append(np.sum(grid == 2))
        population_data[current_disease]["dead"].append(np.sum(grid == 3))
        population_data[current_disease]["vaccinated"].append(np.sum(grid == 4))
    
    build_graph(grid, current_disease if current_disease else "COVID-19")
    
    # Draw grid boxes
    for i in range(GRID_NUMBER):
        for j in range(GRID_NUMBER):
            if grid[i, j] == 0:
                color = WHITE  # Susceptible
            elif grid[i, j] == 1:
                color = DISEASES[current_disease]["color"] if current_disease else WHITE  # Infected
            elif grid[i, j] == 2:
                color = GREEN  # Immune
            elif grid[i, j] == 3:
                color = BLACK  # Dead
            elif grid[i, j] == 4:
                color = DODGER_BLUE  # Vaccinated
            pygame.draw.rect(screen, color, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # Draw the graph
    graph_surface = draw_graph()
    screen.blit(graph_surface, (WIDTH - 10, 0))  # Display the graph next to the grid
    
    pygame.display.flip()
    clock.tick(FPS)

# Save final grid state
save_grid_to_csv(grid, "final_state.csv")

# Plot population vs. time for each disease
for disease, data in population_data.items():
    plt.figure()
    plt.plot(data["susceptible"], label="Susceptible")
    plt.plot(data["infected"], label="Infected")
    plt.plot(data["immune"], label="Immune")
    plt.plot(data["dead"], label="Dead")
    plt.plot(data["vaccinated"], label="Vaccinated")
    plt.title(f"Population vs. Time for {disease}")
    plt.xlabel("Time Steps")
    plt.ylabel("Population")
    plt.legend()
    plt.show()

pygame.quit()