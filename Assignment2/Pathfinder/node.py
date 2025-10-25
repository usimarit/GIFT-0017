class Node:
  def __init__(self, grid, x, y):
    self.x = x
    self.y = y
    self.grid = grid
    self.type = 'road'
    self.g_score = float('inf')
    self.f_score = float('inf')

  def get_neighbors(self):
    # Collection of arrays representing the x and y displacement
    rows = len(self.grid)
    cols = len(self.grid[0])
    directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    neighbors = []
    for direction in directions:
      neighbor_x = self.x + direction[0]
      neighbor_y = self.y + direction[1]
      if neighbor_x >= 0 and neighbor_y >= 0 and neighbor_x < cols and neighbor_y < rows:
        neighbors.append(self.grid[neighbor_y][neighbor_x])
    return neighbors