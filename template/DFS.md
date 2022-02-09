# DFS  deepth first search 

## 基本知识
1. Graph - 某种特征的路径或者长度
2. tree/graph - 遍历
3. tree/graph - 所有方案
4. 排列组合
5. 递归题目都可以用迭代来写，但是实现起来非常麻烦

## 使用条件
1. 找出满足某个条件的所有方案 (90%)
2. 二叉树binary tree的问题 (90%)
3. 组合问题 (95%): 
1) 问题模型: 求出所有满足条件的组合
2) 判断条件: 组合中的元素是顺序无关的

4. 排列问题 （95%）
1) 问题模型: 求出所有满足条件的“排列”
2) 判断条件：组合中的元素是顺序“相关”的

## 不要使用DFS的场景
1. 连通块问题（一定要用BFS， 否则stackoverflow）
2. 拓扑排序 （一定要用BFS， 否则stackoverflow）
3. 一切能用BFS可以解决的问题

## complexity： O(方案个数 * 构造方案的时间)
1. 树的遍历 o(n)
2. 排列问题: o(n! x n)
3. 组合问题： o(2^n x n)

## template 
1. 需要记录路径，不需要返回值
2. 不需要记录路径，需要记录某些特征的返回值

``` python

def dfs(参数列表):
    if 递归出口:
        记录答案
    return

    for 所有的拆解可能性：
        修改所有的参数
        dfs(参数列表)
        还原所有被修改过的参数

    return something 如果需要的话，很多时候不需要return 值除了分治的写法！

```

## 二叉树的模板，前中后序的递归，迭代模板！

### 前序遍历 preorder - 中左右
```python 
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        
        result = []
        def traversal(node):
            if not node:
                return 
            result.append(node.val) # middle 
            traversal(node.left) # left
            traversal(node.right) # right
            return result
        
        traversal(root)
        return result 
```

### 中序遍历 inorder - 左中右
```python 
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        
        result = []
        def traversal(node):
            if not node:
                return 
            traversal(node.left) # left
            result.append(node.val) # middle 
            traversal(node.right) # right
            return result
        
        traversal(root)
        return result 
```

### 后序遍历 postorder - 左右中
```python 
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        
        result = []
        def traversal(node):
            if not node:
                return 
            traversal(node.left) # left
            traversal(node.right) # right
            result.append(node.val) # middle 
            return result
        
        traversal(root)
        return result 
```


## 二叉树的迭代统一模板

### 前序遍历 preorder - 中左右
```python 
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        result = []
        st= []
        if root:
            st.append(root)
        while st:
            node = st.pop()
            if node != None:
                if node.right: #右
                    st.append(node.right)
                if node.left: #左
                    st.append(node.left)
                st.append(node) #中
                st.append(None)
            else:
                node = st.pop()
                result.append(node.val)
        return result
```

### 中序遍历 inorder - 左中右
```python 
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        result = []
        st = []
        if root:
            st.append(root)
        while st:
            node = st.pop()
            if node != None:
                if node.right: #添加右节点（空节点不入栈）
                    st.append(node.right)
                
                st.append(node) #添加中节点
                st.append(None) #中节点访问过，但是还没有处理，加入空节点做为标记。
                
                if node.left: #添加左节点（空节点不入栈）
                    st.append(node.left)
            else: #只有遇到空节点的时候，才将下一个节点放进结果集
                node = st.pop() #重新取出栈中元素
                result.append(node.val) #加入到结果集
        return result
```

### 后序遍历 postorder - 左右中
```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        result = []
        st = []
        if root:
            st.append(root)
        while st:
            node = st.pop()
            if node != None:
                st.append(node) #中
                st.append(None)
                
                if node.right: #右
                    st.append(node.right)
                if node.left: #左
                    st.append(node.left)
            else:
                node = st.pop()
                result.append(node.val)
        return result
```


--- 

## 基于tree的DFS 

## Examples：  863. All Nodes Distance K in Binary Tree

Given the root of a binary tree, the value of a target node target, and an integer k, return an array of the values of all nodes that have a distance k from the target node.

You can return the answer in any order.
```
Input: root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, k = 2
Output: [7,4,1]
Explanation: The nodes that are a distance 2 from the target node (with value 5) have values 7, 4, and 1.
```

>Solution 
```python  
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:

# idea - using dfs to know the par node 

        def dfs(node, par=None):
            if node:
                node.par = par 
                dfs(node.left, node)
                dfs(node.right, node)

        dfs(root)

        # bfs 
        from collections import deque
        queue = deque()
        queue.append((target, 0)) # 从target node开始bfs 这个要记住，不是root
        visited = {target}

        res = []
        while queue:
            if queue[0][1] == k: # distacne == k, 那么是遍历现在queue中所有的node.val
                for node, d in queue:
                    res.append(node.val)
                return res # 别忘了返回值！！！！！

            # 之后pop
            node,dis = queue.popleft()
            for neighbor in (node.left, node.right, node.par): # 比图简单多了
                if not neighbor or neighbor in visited: # 模板写法
                    continue 
                queue.append((neighbor, dis + 1))
                visited.add(neighbor) # 要用add，而不是append

        return []


```

