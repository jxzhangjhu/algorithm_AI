# BFS  brandth first search 

## 三种主要问题和使用场景
1. Graph连通块问题 connected component. a) 通过一个点找到图中连通的所有点 b)非递归的方式找到所有方案 
2. tree/graph 分层遍历 （level order traversal） a) 图的层次遍历 b) 简单图最短路径 simple graph shortest path 
3. 拓扑排序 topological sorting a) 任意拓扑排序，b) 是否有拓扑排序 c) 字典序的最小的拓扑排序 d) 是否 唯一拓扑排序

## 使用条件
1. 拓扑排序 (100%)
2. 出现连通块 connected component 关键词 （100%）
3. 分层遍历 （100%） - 代码回想录那个模板不错
4. 简单图最短路径 （100%）
5. 给定一个变换规则，从初始状态变到终止状态 最少几步（100%）

## complexity
1. time - o(n+m) n是node 点数，m是edge边数，对于tree怎么办？
2. 基于tree的bfs
3. 基于graph的bfs 

## template 

1. general template 
```python
from collections import deque
def bfs(start_node):
    
    # BFS 必须用队列queue， 别用stack
    # distacne(dict) 有2个作用： 1） 记录一个点是否被丢进过队列了，避免重复访问 2） 记录start_node 到其他所有节点的距离
    # 如果只求连通性的话，可以换成set就行
    # node做key 的时候比较的是内存地址
    queue = deque([start_node]) 
    distance = {start_node:0}

    # while队列不空，不停的从队列里拿出一个点，拓展邻居节点放到队列中 
    while queue:
        node = queue.popleft()

        # 如果有明确的终点可以在这里加对终点的判断

        if node is 终点:
            break or return something 

        for neighboor in node.get_neighbors():
            if neighbor in distance:
                continue 
            queue.append(neighbor)
            distance[neighbor] = distance[node] + 1 

            # 如果需要返回所有点离起点的距离，就return hashmap
            return distance 

            # 如果需要返回所有连通的节点，就return hashmap 里的所有点

            return distance.keys()

            #如果需要返回离终点的最短距离
            return distance[end_node]

```

2. 拓扑排序 topological sorting 

```python 
def get_indegress(nodes):
    counter = {node:0 for node in nodes}

    for node in nodes:
        for neighbor in node.get_neighbors()
            counter[neighbor] += 1
    return counter

    def topological_sort(nodes):
        # 统计入度
        indegrees = get_indegrees(nodes)
        # 所有入度为0的点都放到队列里
        queue = collections.deque(
            [node for node in nodes if indegress[node] == 0]
            )

        # 用BFS算法一个个的把点从图里挖出来
        topo_order = []
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for neighbor in node.get_neighbors():
                indegrees[neighbor] -= 1
                if indegrees[neighbor] == 0:
                    queue.append(neighbor)
        # 判断是否有循环依赖 

        if len(topo_order) != len(nodes):
            return 有循环依赖（环），没有拓扑序

        return topo_order 

```


## Examples 

### 200. Number of Islands
https://leetcode.com/problems/number-of-islands/

Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.


Example 1:
```
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
```
Example 2:
```
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
```

>Solution

```python
# 定义2d 矩阵的方向, x, y 方向分别 +1 -1  需要定义成全局变量！
direct = [(1, 0), (0, -1), (-1, 0), (0, 1)] 
from collections import deque

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        
        # BFS的核心其实是遍历，然后通过遍历的过程，记录想要得到的信息，本质思想的是通过queue + for loop 逐行扫描记录来实现
        # 对于这道题，就是line by line 遍历，同时记录遍历过程中的特点 - 是否有相邻的1， 如果1 有相邻的，那么岛数不变，否则+1
        
        if not grid or not grid[0]:
            return 0
        
        island_num = 0 
        # visted 很重要记录是否被BFS过，如果之前已经被BFS过，不应再次bfs，有点减枝的效果，pruning tree的想法，其实不visited也可以，暴力干
        visited = set() # 不用hash map 因为只需要看是否存在过
        
        for i in range(len(grid)): # 行
            for j in range(len(grid[0])): # 列
                # 如果是0， 海洋，没必要bfs
                # 如果已经被visited， 无需再做bfs
                # print(grid[i][j]=='1') # 这个很重要，grid[i][j]==1 return false， 因为grid[i][j] =‘1’ 是字符串，不是int，要小心
                if grid[i][j] == '1' and (i, j) not in visited: 
                    self.bfs(grid, i, j, visited)
                    island_num += 1
        
        return island_num
    
    def bfs(self, grid, x, y, visited):
        queue = deque([(x, y)]) # 如果是tree的话，这里是node
        visited.add((x, y))

        while queue:
            x, y = queue.popleft()
            # print((x,y))
            for delta_x, delta_y in direct:
                next_x = x + delta_x
                next_y = y + delta_y
                if self.is_valid(grid, next_x, next_y, visited):
                    queue.append((next_x, next_y)) # queue uses append 不是list 
                    visited.add((next_x, next_y)) # set use add
                else:
                    continue 
                
    def is_valid(self, grid, x, y, visited):
        # 是否是有效的点需要bfs - 有几种情况不需要, return false
        # 1） 点在matrix 外部，就是边界点
        # 2） 点已经被visited
        
        n, m = len(grid), len(grid[0])
        if (0 <= x < n and 0 <= y < m) and grid[x][y] == '1' and (x, y) not in visited:
            return True 
        else:
            return False 
        
        # 注意和九章那个区别，这里是‘1’ and ‘0’, 字符串不能直接return true or false 

#  Complexity Analysis

# Time complexity : O(M \times N)O(M×N) where MM is the number of rows and NN is the number of columns.

# Space complexity : O(min(M, N))O(min(M,N)) because in worst case where the grid is filled with lands, the size of queue can grow up to min(M,NM,N).
         
``` 

### 133. Clone Graph
https://leetcode.com/problems/clone-graph/

Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.
```
class Node {
    public int val;
    public List<Node> neighbors;
}
```
Test case format:

For simplicity, each node's value is the same as the node's index (1-indexed). For example, the first node with val == 1, the second node with val == 2, and so on. The graph is represented in the test case using an adjacency list.

An adjacency list is a collection of unordered lists used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.

The given node will always be the first node with val = 1. You must return the copy of the given node as a reference to the cloned graph.

```
Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
Explanation: There are 4 nodes in the graph.
1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
```

>Solution 
```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        
        """
        :type node: Node
        :rtype: Node
        """
        # 有一些操作不熟悉，但是思路是类似层级遍历！ 
        if not node:
            return node

        # Dictionary to save the visited node and it's respective clone
        # as key and value respectively. This helps to avoid cycles.
        visited = {}

        # Put the first node in the 
        print(node.val) # 1
        print(node.neighbors[0].val) # 2
        print(node.neighbors[1].val) # 4
        
        queue = deque([node])
        # Clone the node and put it in the visited dictionary.
        visited[node] = Node(node.val, []) # 这个很重要，记录各种想要的results

        # Start BFS traversal
        while queue:
            # Pop a node say "n" from the from the front of the queue.
            n = queue.popleft()
            print(n.val)
            # Iterate through all the neighbors of the node
            for neighbor in n.neighbors: # 宽度优先，就是有几个，这里是2个neighbors
                # print(neighbor) 
                if neighbor not in visited:
                    # Clone the neighbor and put in the visited, if not present already
                    visited[neighbor] = Node(neighbor.val, [])
                    # Add the newly encountered node to the queue.
                    queue.append(neighbor)
                # Add the clone of the neighbor to the neighbors of the clone node "n".
                visited[n].neighbors.append(visited[neighbor])

        # Return the clone of the node from visited.
        return visited[node]
        
# Complexity Analysis

# Time Complexity : O(N + M), where N is a number of nodes (vertices) and M is a number of edges. 注意不是NxM

# Space Complexity : O(N). This space is occupied by the visited dictionary and in addition to that, space would also be occupied by the queue since we are adopting the BFS approach here. The space occupied by the queue would be equal to O(W) where W is the width of the graph. Overall, the space complexity would be O(N).
```
   

### Example: 490. The Maze
https://leetcode.com/problems/the-maze/


There is a ball in a maze with empty spaces (represented as 0) and walls (represented as 1). The ball can go through the empty spaces by rolling up, down, left or right, but it won't stop rolling until hitting a wall. When the ball stops, it could choose the next direction.

Given the m x n maze, the ball's start position and the destination, where start = [startrow, startcol] and destination = [destinationrow, destinationcol], return true if the ball can stop at the destination, otherwise return false.

You may assume that the borders of the maze are all walls (see examples).     

```
Input: maze = [[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]], start = [0,4], destination = [4,4]
Output: true
Explanation: One possible way is : left -> down -> left -> down -> right -> down -> right.
```

>Solution
```python

from collections import deque
# define the global direction 
dirct = [(1, 0), (-1, 0), (0, 1), (0, -1)]
class Solution:
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        queue = deque([(start[0], start[1])]) 
        visited = set()
        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue 
            if curr == (destination[0], destination[1]):
                return True
            visited.add(curr)
            for neigh in self.neighbor_count(curr[0],curr[1], maze):
                queue.append(neigh)
                # print(queue)
        return False 

    def neighbor_count(self, x, y, maze):
        len_x = len(maze)
        len_y = len(maze[0])
        # return neighbors
        neighbors = [] 
        visited = set()
        visited.add((x,y)) # add the current (x,y)， 防止当前的被加进去？
        for dx, dy in dirct:
            curr_x, curr_y = x, y
            # 判断条件 1) 不超过边界，2） 不是wall，也就是值是0
            while 0 <= curr_x + dx < len_x and 0 <= curr_y + dy < len_y and maze[curr_x + dx][curr_y + dy] == 0: 
                curr_x += dx  # 直接走到头
                curr_y += dy  # 不是走一步
            if (curr_x, curr_y) not in visited:
                neighbors.append((curr_x, curr_y))
        return neighbors 

```

                    
        
        
        