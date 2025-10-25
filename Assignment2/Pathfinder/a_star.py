import heapq
from collections import deque

def manhattan_distance(node1, node2):
    return abs(node1.x - node2.x) + abs(node1.y - node2.y)

def a_star(start, end):
    """
    A* 寻路算法实现
    返回: (path, visit_history) 元组，path是找到的路径，visit_history是访问过的节点列表
    """
    # 初始化开放列表（优先队列）和关闭列表（集合）
    open_set = []
    closed_set = set()
    visit_history = []  # 记录访问顺序
    
    # 用来重建路径的字典
    came_from = {}
    
    # 初始化起点
    start.g_score = 0
    start.f_score = manhattan_distance(start, end)
    heapq.heappush(open_set, (start.f_score, id(start), start))
    
    while open_set:
        # 获取f值最小的节点
        current = heapq.heappop(open_set)[2]
        visit_history.append(current)
        
        if current == end:
            # 找到路径，重建并返回
            path = reconstruct_path(came_from, current)
            return path, visit_history
            
        closed_set.add(current)
        
        # 检查所有邻居节点
        for neighbor in current.get_neighbors():
            # 跳过墙壁和已经在关闭列表中的节点
            if neighbor in closed_set or neighbor.type == 'wall':
                continue
                
            # 计算经过当前节点到达邻居节点的g值
            tentative_g_score = current.g_score + 1  # 假设每步代价为1
            
            # 如果这是一个新节点或者找到了更好的路径
            if neighbor not in [item[2] for item in open_set] or tentative_g_score < neighbor.g_score:
                # 更新路径信息
                came_from[neighbor] = current
                neighbor.g_score = tentative_g_score
                neighbor.f_score = neighbor.g_score + manhattan_distance(neighbor, end)
                
                # 添加到开放列表
                if neighbor not in [item[2] for item in open_set]:
                    heapq.heappush(open_set, (neighbor.f_score, id(neighbor), neighbor))


    # 没有找到路径
    return None, visit_history


def bfs(start, end):
    queue = deque([start])
    visited = set()
    came_from = {}
    visit_history = []

    while queue:
        current = queue.popleft()
        visit_history.append(current)
        if current == end:
            return reconstruct_path(came_from, current), visit_history

        visited.add(current)

        for neighbor in current.get_neighbors():
            if neighbor not in visited  and neighbor not in queue and neighbor.type != 'wall':
                queue.append(neighbor)
                came_from[neighbor] = current

    return None  # No path found

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path