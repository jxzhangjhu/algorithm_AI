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

## 基于tree的DFS 

## Examples 


