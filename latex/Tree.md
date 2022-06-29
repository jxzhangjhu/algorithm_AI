# Tree 

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

✅✅✅ Tree recursive/DFS ✅✅✅ 

### 235. Lowest Common Ancestor of a Binary Search Tree
Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”
```
Example 1:
Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.

Example 2:
Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
Output: 2
Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.

Example 3:
Input: root = [2,1], p = 2, q = 1
Output: 2
```
> recursive 经典题，找公共祖先
```python
# Definition for a binary tree node. # 这段需要自己写！
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        while True:
            if root.val > p.val and root.val > q.val:
                root = root.left
            elif root.val < p.val and root.val < q.val:
                root = root.right
            else:
                return root
``` 


### 863. All Nodes Distance K in Binary Tree

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

✅✅✅ Tree BFS、层序遍历 ✅✅✅ 


### 226. Invert Binary Tree 
Given the root of a binary tree, invert the tree, and return its root.
```
Example 1:
Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]
Example 2:
Input: root = [2,1,3]
Output: [2,3,1]
Example 3:
Input: root = []
Output: []
```
> 经典tree-based BFS题，大概有类似15道左右的都是可以用层序模板解决！
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:  # # 这个地方不能写[]，不是0，or -1 而是nothing
            # return None # ok,  
            # return  # ok
            return root # ok
        else:
            queue = collections.deque([root])
            print(queue)            
        while queue:
            n = len(queue)
            for i in range(n):
                curnode = queue.popleft()
                curnode.left, curnode.right = curnode.right, curnode.left
                if curnode.left:
                    queue.append(curnode.left)
                if curnode.right:
                    queue.append(curnode.right)
        return root
```



### 110. Balanced Binary Tree 
Given a binary tree, determine if it is height-balanced. For this problem, a height-balanced binary tree is defined as: a binary tree in which the left and right subtrees of every node differ in height by no more than 1.
```
Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: true
Example 2:
Input: root = [1,2,2,3,3,null,null,4,4]
Output: false
Example 3:
Input: root = []
Output: true
```
> 这个题挺好的，先用recursive + bfs 层序遍历
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        st = []
        if not root:
            return True
        st.append(root)
        while st:
            node = st.pop() #中
            if abs(self.getDepth(node.left) - self.getDepth(node.right)) > 1:
                return False
            if node.right:
                st.append(node.right) #右（空节点不入栈）
            if node.left:
                st.append(node.left) #左（空节点不入栈）
        return True
    
    # 这个是get depth using BFS层序遍历
    def getDepth(self, cur):
        st = []
        if cur:
            st.append(cur)
        depth = 0
        result = 0
        while st:
            node = st.pop()
            if node:
                st.append(node) #中
                st.append(None)
                depth += 1
                if node.right: st.append(node.right) #右
                if node.left: st.append(node.left) #左
            else:
                node = st.pop()
                depth -= 1
            result = max(result, depth)
        return result        

```

