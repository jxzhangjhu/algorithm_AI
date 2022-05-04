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

## 记忆化搜索！ memo 
1. 在函数返回前，记录函数的返回结果，再一次以同样的参数访问函数直接返回记录下的结果
2. 递归返回时同时记录下已访问过的节点特征，可以将指数级别的DFS降低到多项式级别
3. 这一类DFS必须有返回值，不可以用排列组合的DFS写法
4. for 循环的dp题目都可以用记忆化搜索的方式写，但是不是所有的记忆化搜索题目都可以用for循环的dp写 

### 三个特点 
1. 函数有返回值
2. 函数返回结果和输入参数相关，和其他全局状态无关
3. 参数列表中传入哈希表或者其他用于记录计算结果的数据结构

### 三种使用场景 - 核心思想是，由大化小，递归，分治思想
1. 求可行性
2. 求方案数
3. 求最值


## DFS 模板
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

## 回溯模板 backtracking - 代码回想录的
```C++
void backtracking(参数) {
    if (终止条件) {
        存放结果;
        return;
    }

    for (选择：本层集合中元素（树中节点孩子的数量就是集合的大小）) {
        处理节点;
        backtracking(路径，选择列表); // 递归
        回溯，撤销处理结果
    }
}
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


### Example 22. Generate Parentheses 很不错的题，知识点，dfs，控制，括号，判断条件，递归
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
Example 1:
```
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
```
Example 2:
```
Input: n = 1
Output: ["()"]
```

>Solution
```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:

# https://leetcode-cn.com/problems/generate-parentheses/solution/sui-ran-bu-shi-zui-xiu-de-dan-zhi-shao-n-0yt3/

#         # first 用递归写出所有的组合在判断是否valid
#         if n <= 0: return 
        
#         res = []
#         def dfs(paths):
#             # 先写递归出口
#             if len(paths) == 2*n:
#                 res.append(paths)
#                 return 
#             # 递归定义， 只有paths 放进去作为参数
#             dfs(paths + '(')
#             dfs(paths + ')')
           
#         dfs('')
#         print(res)
#         return res 
    
    # 这其中很多都不对，因为是暴力枚举，所以所有情况都在里面了！ 
    # 改进方法： 要判断左右括号的数量, 两者要相等才行
        
        if n <= 0: return 
        res = []
        def dfs(paths, num_left, num_right):
            if num_left > n or num_left < num_right: # 这个有讲究，关键是括号的处理
                return 

            # if num_left > n or num_left < num_right:
            # if num_left > n or num_left < num_right: # 这个有讲究，关键是括号的处理
            # if num_left > n or num_left != num_right: # 这个就过不去，因为不等于是可以的，因为要左括号在前，然后才是右括号，如果改成left > right 也不行    
                
                # return 
            # 先写递归出口
            if len(paths) == n * 2:
                res.append(paths)
                return 
            # 递归定义， 只有paths 放进去作为参数
            dfs(paths + '(', num_left + 1, num_right)
            dfs(paths + ')', num_left, num_right + 1)
            
           
        dfs('', 0, 0)

        return res 
    
        # way 1 - 暴力解法，看不太懂！ 
#         def generate(A): 
#             if len(A) == 2*n:
#                 if valid(A):
#                     ans.append("".join(A))
#             else:
 
#                 A.append('(')
#                 generate(A)
#                 A.pop()
#                 A.append(')')
#                 generate(A)
#                 A.pop()

#         def valid(A):
#             bal = 0
#             for c in A:
#                 if c == '(': bal += 1
#                 else: bal -= 1
#                 if bal < 0: return False
#             return bal == 0

#         ans = []
#         generate([])
#         return ans
```

--- 
## memo 方法的例题
### Example 139. Word Break 
Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

Example 1:
```
Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
```
Example 2:
```
Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.
```
Example 3:
```
Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: false
```

> Solution 
1. DFS + memo 
下面这个例子中，start 指针代表了节点的状态，可以看到，做了大量重复计算：
用一个数组，存储计算的结果，数组索引为指针位置，值为计算的结果。下次遇到相同的子问题，直接返回命中的缓存值，就不用调重复的递归。
***memo*** 是用一个数组，来存储dfs的返回值，dfs一定有返回值，但可能是true or false，这样把结果存在memo中，不用反复的调用，如果已经调用了就返回memo的结果！

```python
# 这个链接非常好，讲的很到位！三种不同的方法！ 
# https://leetcode-cn.com/problems/word-break/solution/shou-hui-tu-jie-san-chong-fang-fa-dfs-bfs-dong-tai/
    
    
# class Solution:
#     def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        
#         # '''排列''' - for loop dp 解法 
#         # dp = [False]*(len(s) + 1)
#         # dp[0] = True
#         # # 遍历背包
#         # for j in range(1, len(s) + 1):
#         #     # 遍历单词
#         #     for word in wordDict:
#         #         if j >= len(word):
#         #             dp[j] = dp[j] or (dp[j - len(word)] and word == s[j - len(word):j])
#         # return dp[len(s)]

#         # bfs 方法 队列实现  - 如果不加visited 控制，那么就是超时！ 
#         queue = collections.deque()
#         queue.append(0)
#         visited = set()  # 初始化记忆
        
#         while queue:
#             i = queue.popleft()
#             if i in visited: # 模板，如果访问过，这一轮循环就跳过，如果没有，加到set中
#                 continue
#             else:
#                 visited.add(i)
            
#             for j in range(i, len(s)):
#                 if s[i:j+1] in wordDict:
#                     if j == len(s)-1:
#                         return True
#                     else:
#                         queue.append(j+1)
#         return False
                
# complexity 
# n is the length of the input string.
# Time complexity : O(n^3) For every starting index, the search can continue till the end of the given string.
# Space complexity : O(n). Queue of at most nn size is needed.


# DFS 方法 + memo
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        
        # 直接dfs 暴力的话，就是超时！
        memo = [None] * len(s)
        def dfs(i): # 以索引i为起始到末尾的字符串能否由字典组成
            if i >= len(s):
                return True             
            if memo[i] != None:
                return memo[i]
            for j in range(i, len(s)):
                if s[i:j+1] in wordDict and dfs(j+1): # 假设dfs(j+1) 也是一个单词，这是递归的核心！有返回值！
                    return True
                    memo[i] = True
            memo[i] = False
            return False
        
        return dfs(0)
            
        # 下面是DFS + memo，怎么用memo，都是存成true or false吗？
        
#         memo = [None]*len(s) # 这个memo就跟visited 类似的效果！

#         # 以索引i为起始到末尾的字符串能否由字典组成
#         def dfs(i):
#             # 长度超过s,返回True(空字符能组成)
#             if i >= len(s): 
#                 return True
#             # 存在以i为起始的递归结果
#             if memo[i] != None:
#                 return memo[i]
#             # 递归
#             for j in range(i,len(s)):
#                 if s[i:j+1] in wordDict and dfs(j+1):
#                     memo[i] = True
#                     return True
#             memo[i] = False
#             return False
        
#         return dfs(0)

# complexity 
# n is the length of the input string.
# Time complexity : O(n^3) For every starting index, the search can continue till the end of the given string.
# Space complexity : O(n). Queue of at most nn size is needed.

## DP 方法 - 后面再做！time o(n^3) and space o(n)

``` 


### Example 241. Different Ways to Add Parentheses
https://leetcode.com/problems/different-ways-to-add-parentheses/ 

Share
Given a string expression of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators. You may return the answer in any order.

Example 1:
```
Input: expression = "2-1-1"
Output: [0,2]
Explanation:
((2-1)-1) = 0 
(2-(1-1)) = 2
```
Example 2:
```
Input: expression = "2*3-4*5"
Output: [-34,-14,-10,-10,10]
Explanation:
(2*(3-(4*5))) = -34 
((2*3)-(4*5)) = -14 
((2*(3-4))*5) = -10 
(2*((3-4)*5)) = -10 
(((2*3)-4)*5) = 10
```
>Solution 
1. 分解子问题就是x o y 考虑到left and right的操作
2. 注意参数的变化，以i为界，考虑左右的变化，同时有2个for loop去遍历，然后要append所有的可能性！
3. 并没有跟（）有任何关系，不要被误导了！

```python
class Solution:
    def diffWaysToCompute(self, expression: str) -> List[int]:
        # 如果只有数字，直接返回
        if expression.isdigit():
            return [int(expression)]

        res = []
        for i, char in enumerate(expression):
            if char in ['+', '-', '*', '/']:
                left = self.diffWaysToCompute(expression[:i])
                right = self.diffWaysToCompute(expression[i+1:])
                for l in left:
                    for r in right:
                        if char == '+':
                            res.append(l + r)
                        if char == '-':
                            res.append(l - r)
                        if char == '*':
                            res.append(l * r)
                        if char == '/':
                            res.append(l / r)
        return res 

# 单独再写dfs比较麻烦，直接用这个函数写完了！
```
