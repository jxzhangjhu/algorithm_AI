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

### Example： 130. Surrounded Regions
https://leetcode.com/problems/surrounded-regions/ 

Given an m x n matrix board containing 'X' and 'O', capture all regions that are 4-directionally surrounded by 'X'.

A region is captured by flipping all 'O's into 'X's in that surrounded region.

>Solution
```python
# 这个题思路其实比较简单，就是操作复杂一些，比如处理边界的，没有用2d 矩阵的方式？
class Solution(object):
    def solve(self, board):        
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        if not board or not board[0]:
            return

        self.ROWS = len(board)
        self.COLS = len(board[0])

        # 单独把所有的边界的都提出来
        # Step 1). retrieve all border cells
        from itertools import product
        borders = list(product(range(self.ROWS), [0, self.COLS-1])) \
                + list(product([0, self.ROWS-1], range(self.COLS)))
        
        print(borders)

        # Step 2). mark the "escaped" cells, with any placeholder, e.g. 'E' - 先处理边界上的，都写成E
        for row, col in borders:
            #self.DFS(board, row, col)
            self.BFS(board, row, col)
            

        # Step 3). flip the captured cells ('O'->'X') and the escaped one ('E'->'O') # 再处理内部的
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if board[r][c] == 'O':   board[r][c] = 'X'  # captured
                elif board[r][c] == 'E': board[r][c] = 'O'  # escaped


    def BFS(self, board, row, col):
        from collections import deque
        queue = deque([(row, col)])
        while queue:
            (row, col) = queue.popleft()
            if board[row][col] != 'O':
                continue
            # mark this cell as escaped
            board[row][col] = 'E'
            # check its neighbor cells
            if col < self.COLS-1: queue.append((row, col+1))
            if row < self.ROWS-1: queue.append((row+1, col))
            if col > 0: queue.append((row, col-1))
            if row > 0: queue.append((row-1, col))    
                      
    # 时间和空间都是o(n)
 
```
      
### Example: 286. Walls and Gates 
https://leetcode.com/problems/walls-and-gates/ 

You are given an m x n grid rooms initialized with these three possible values.

-1 A wall or an obstacle.
0 A gate.
INF Infinity means an empty room. We use the value 231 - 1 = 2147483647 to represent INF as you may assume that the distance to a gate is less than 2147483647.
Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.
> Solution

```python 
### 这个题思路比较另类，实际上用gate去遍历所有的空， 
from collections import deque
class Solution:
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        # initialization 
        m, n = len(rooms), len(rooms[0])

        # 遇到2D matrix 就用这个neighbor，明确哪些neighbor要记录
        def neighbor(x,y):
            neighbors = []
            direct = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dx, dy in direct: 
                new_x = x + dx 
                new_y = y + dy
                if 0 <= new_x < m and 0 <= new_y < n and rooms[new_x][new_y] == 2147483647: # 加这个判断条件
                    neighbors.append((new_x, new_y))
            return neighbors

        #  build a queue and add all gate 
        gate = [] 
        for i in range(m):
            for j in range(n):
                if rooms[i][j] == 0:
                    gate.append((i,j))

        # bfs 
        queue = deque(gate)
        distance = 0
        visited = set(gate) # 需要记录，否则不对，会走重复的
        while queue:
            for _ in range(len(queue)): # 需要遍历这个，因为所有的门一起开始移动，one by one 而不是在一个点的滑动 
                x, y = queue.popleft()
                rooms[x][y] = distance
                for i, j in neighbor(x, y):
                    if rooms[i][j] == -1 or (i, j) in visited:
                        continue 
                    queue.append((i,j))
                    visited.add((i,j))

            distance += 1

        # time o(mxn), space o(mxn) 就是遍历所有的点

```

### Example: 1091. Shortest Path in Binary Matrix 最短路径问题
https://leetcode.com/problems/shortest-path-in-binary-matrix/ 


Given an n x n binary matrix grid, return the length of the shortest clear path in the matrix. If there is no clear path, return -1.

A clear path in a binary matrix is a path from the top-left cell (i.e., (0, 0)) to the bottom-right cell (i.e., (n - 1, n - 1)) such that:

All the visited cells of the path are 0.
All the adjacent cells of the path are 8-directionally connected (i.e., they are different and they share an edge or a corner).
The length of a clear path is the number of visited cells of this path.

>Solution 

```python 
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:

        # 如何处理这种最短路径的问题？ 
        # 初始化queue的问题中，很多是把distance 加到queue里面
        
        n = len(grid)
        res = float('inf')
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] # 8个方向的
        queue = deque([(0 , 0, 1)]) # 这里加个1是distance 的起始点
        
        # 加 # 就是为了避免重复，实际上是用visited 代替的很多case 
        visited = set()
        while queue:
            row, col, step = queue.popleft()
            
            # if not (0 <= row < n) or not (0 <= col < n) or grid[row][col] == 1 or grid[row][col] == '#':
            #     continue
            # grid[row][col] = '#'
            
            if not (0 <= row < n) or not (0 <= col < n) or grid[row][col] == 1 or (row, col) in visited: 
                continue
            visited.add((row, col))

            for x, y in directions:
                newRow = row + x
                newCol = col + y
                queue.append((newRow, newCol, step + 1))
                
            if row == n - 1 and col == n - 1:
                res = min(res, step)
                continue
        
        return res if res != float('inf') else -1 # 这个output也要注意，判断一下 res 是不是动了，否则就是inf，那么就是-1 

```

### Example: 542. 01 Matrix 
https://leetcode.com/problems/01-matrix/ 

Given an m x n binary matrix mat, return the distance of the nearest 0 for each cell.

The distance between two adjacent cells is 1 

>Solution 

```python
class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        m, n = len(matrix), len(matrix[0])
        dist = [[0] * n for _ in range(m)]
        print(dist)
        zeroes_pos = [(i, j) for i in range(m) for j in range(n) if matrix[i][j] == 0]
        print(zeroes_pos)
        # 将所有的 0 添加进初始队列中
        q = collections.deque(zeroes_pos)
        seen = set(zeroes_pos) # 这步很关键，所有的0都进去了，只需要扫其他的1， 即便是最短path，但不需要用min来判断
        
        # 广度优先搜索
        while q:
            i, j = q.popleft()
            for ni, nj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if 0 <= ni < m and 0 <= nj < n and (ni, nj) not in seen:
                    dist[ni][nj] = dist[i][j] + 1
                    q.append((ni, nj))
                    seen.add((ni, nj))
        
        return dist

# 复杂度分析

# 时间复杂度：O(rc)O(rc)，其中 rr 为矩阵行数，cc 为矩阵列数，即矩阵元素个数。广度优先搜索中每个位置最多只会被加入队列一次，因此只需要 O(rc)O(rc) 的时间复杂度。

# 空间复杂度：O(rc)O(rc)，其中 rr 为矩阵行数，cc 为矩阵列数，即矩阵元素个数。除答案数组外，最坏情况下矩阵里所有元素都为 00，全部被加入队列中，此时需要 O(rc)O(rc) 的空间复杂度。


``` 

### Example: 314. Binary Tree Vertical Order Traversal 
https://leetcode.com/problems/binary-tree-vertical-order-traversal/ 

Given the root of a binary tree, return the vertical order traversal of its nodes' values. (i.e., from top to bottom, column by column).

If two nodes are in the same row and column, the order should be from left to right.


<!-- Approach 1. Breadth-first search / level-order traversal with sorting
Time complexity: O(n*logn)
Space complexity: O(n)
The requirement is that we traverse the tree from top to bottom and left to right. Right away, this should make one think of breadth-first search / level order traversal because that has the same traversal order. Because we need each column slice of nodes to be its own list in the return output/list, we need to keep track of each node's offset from the root node. Since the root node is the center of our tree, we assign it an offset of zero. Thus, the left child has an offset of -1 and each left child will have an offset one less than its parent. Similarly, the right child has an offset of +1 and each right child will have an offset one more than its parent. See the following image: -->

```python
class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        offset_map = {} # 用dict map
        queue = deque([(root, 0)] if root else [])

        while queue:
            node, offset = queue.popleft()

            if offset not in offset_map:
                offset_map[offset] = [node.val]
            else:
                offset_map[offset].append(node.val)

            if node.left:
                queue.append((node.left, offset - 1)) # left 就减一层

            if node.right:
                queue.append((node.right, offset + 1)) # 加一层

        return [v for k, v in sorted(offset_map.items())]

```
<!-- Approach 2: breadth-first search without sorting
Time complexity: O(n)
Space complexity: O(n)
Due to the requirement that we traverse left to right, we needed to sort the offsets/keys in ascending order for Approach 1, thus increasing our time complexity to O(n*logn). To not have to do this, we can instead key track of the minimum and maximum offsets and then just iterate through the range in O(n) time (the range would be of the size if we had a skewed tree where each node either had only a left child or only a right child, like a linked list).
 -->
```python
class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        offset_map = {}
        queue = deque([(root, 0)])
        min_offset, max_offset = 0, 0

        while queue:
            node, offset = queue.popleft()

            if offset < min_offset:
                min_offset = offset

            if offset > max_offset:
                max_offset = offset

            if offset not in offset_map:
                offset_map[offset] = [node.val]
            else:
                offset_map[offset].append(node.val)

            if node.left:
                queue.append((node.left, offset - 1))

            if node.right:
                queue.append((node.right, offset + 1))

        return [offset_map[offset] for offset in range(min_offset, max_offset + 1)]
        
```        


### Example: 207. Course Schedule 拓扑排序

There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return true if you can finish all courses. Otherwise, return false.


Example 1:
```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0. So it is possible.
```
Example 2:
```
Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.
```

>Solution
```python

# https://leetcode-cn.com/problems/course-schedule/solution/bao-mu-shi-ti-jie-shou-ba-shou-da-tong-tuo-bu-pai-/ 这个link讲的特别好！  
# class Solution:
#     def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        
#         # 新建一个字典，存储有向图
#         edges = collections.defaultdict(list)
#         print(edges)
#         # 存储每个节点的入度
#         indeg = [0] * numCourses
#         # 字典值的填充和入度的统计
#         for info in prerequisites:
#             edges[info[1]].append(info[0])
#             print(edges)
#             indeg[info[0]] += 1
#         # 把入度为0的课程放入到队列
#         queue = collections.deque([u for u in range(numCourses) if indeg[u] == 0])
#         result = 0

#         while queue:
#             result += 1
#             u = queue.popleft()
#             for v in edges[u]:
#                 indeg[v] -= 1
#                 if indeg[v] == 0:
#                     queue.append(v)

#         return result == numCourses # 已学习课程的数量和input课程数量


class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        from collections import defaultdict
        from collections import deque
        # 入度数组(列表，保存所有课程的依赖课程总数)
        in_degree_list = [0] * numCourses
        # 关系表(字典，保存所有课程与依赖课程的关系)
        relation_dict = defaultdict(list)
        for i in prerequisites:
            # 保存课程初始入度值
            in_degree_list[i[0]] += 1
            # 添加依赖它的后续课程
            relation_dict[i[1]].append(i[0])
        queue = deque()
        for i in range(len(in_degree_list)):
            # 入度为0的课程入列
            if in_degree_list[i] == 0:
                queue.append(i)
        # 队列只存储入度为0的课程，也就是可以直接选修的课程
        while queue:
            current = queue.popleft()
            # 选修课程-1
            numCourses -= 1
            relation_list = relation_dict[current]
            # 如果有依赖此课程的后续课程则更新入度
            if relation_list:
                for i in relation_list:
                    in_degree_list[i] -= 1
                    # 后续课程除去当前课程无其他依赖课程则丢入队列
                    if in_degree_list[i] == 0:
                        queue.append(i)
        return numCourses == 0
```
