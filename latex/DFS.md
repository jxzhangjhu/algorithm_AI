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




✅✅✅ backtracking ✅✅✅   


### 39. Combination Sum https://leetcode.com/problems/combination-sum/
Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.
 
```
Example 1:

Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.
Example 2:

Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]
Example 3:

Input: candidates = [2], target = 1
Output: []
```
```python
# class Solution:
#     def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        
class Solution:
    def __init__(self):
        self.path = []
        self.paths = []

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        '''
        因为本题没有组合数量限制，所以只要元素总和大于target就算结束
        '''
        self.path.clear()
        self.paths.clear()

        # 为了剪枝需要提前进行排序
        candidates.sort()
        self.backtracking(candidates, target, 0, 0)
        return self.paths

    def backtracking(self, candidates: List[int], target: int, sum_: int, start_index: int) -> None:
        # Base Case
        if sum_ == target:
            self.paths.append(self.path[:]) # 因为是shallow copy，所以不能直接传入self.path
            return
        # 单层递归逻辑 
        # 如果本层 sum + condidates[i] > target，就提前结束遍历，剪枝
        for i in range(start_index, len(candidates)):
            if sum_ + candidates[i] > target: 
                return 
            sum_ += candidates[i]
            self.path.append(candidates[i])
            self.backtracking(candidates, target, sum_, i)  # 因为无限制重复选取，所以不是i-1
            sum_ -= candidates[i]   # 回溯
            self.path.pop()        # 回溯
``` 


### 40. Combination Sum II 
Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.

Each number in candidates may only be used once in the combination.

Note: The solution set must not contain duplicate combinations.
```
Example 1:

Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
Example 2:

Input: candidates = [2,5,2,1,2], target = 5
Output: 
[
[1,2,2],
[5]
]
```
```python
# class Solution:
#     def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
       
class Solution:
    def __init__(self):
        self.paths = []
        self.path = []
        self.used = []

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        '''
        类似于求三数之和，求四数之和，为了避免重复组合，需要提前进行数组排序
        本题需要使用used，用来标记区别同一树层的元素使用重复情况：注意区分递归纵向遍历遇到的重复元素，和for循环遇到的重复元素，这两者的区别
        '''
        self.paths.clear()
        self.path.clear()
        self.usage_list = [False] * len(candidates)
        # 必须提前进行数组排序，避免重复
        candidates.sort()
        self.backtracking(candidates, target, 0, 0)
        return self.paths

    def backtracking(self, candidates: List[int], target: int, sum_: int, start_index: int) -> None:
        # Base Case
        if sum_ == target:
            self.paths.append(self.path[:])
            return
        
        # 单层递归逻辑
        for i in range(start_index, len(candidates)):
            # 剪枝，同39.组合总和
            if sum_ + candidates[i] > target:
                return
            
            # 检查同一树层是否出现曾经使用过的相同元素
            # 若数组中前后元素值相同，但前者却未被使用(used == False)，说明是for loop中的同一树层的相同元素情况
            if i > 0 and candidates[i] == candidates[i-1] and self.usage_list[i-1] == False:
                continue

            sum_ += candidates[i]
            self.path.append(candidates[i])
            self.usage_list[i] = True
            self.backtracking(candidates, target, sum_, i+1)
            self.usage_list[i] = False  # 回溯，为了下一轮for loop
            self.path.pop()             # 回溯，为了下一轮for loop
            sum_ -= candidates[i]       # 回溯，为了下一轮for loop
``` 

### 216. Combination Sum III 
Find all valid combinations of k numbers that sum up to n such that the following conditions are true:

Only numbers 1 through 9 are used.
Each number is used at most once.
Return a list of all possible valid combinations. The list must not contain the same combination twice, and the combinations may be returned in any order.
```
Example 1:

Input: k = 3, n = 7
Output: [[1,2,4]]
Explanation:
1 + 2 + 4 = 7
There are no other valid combinations.
Example 2:

Input: k = 3, n = 9
Output: [[1,2,6],[1,3,5],[2,3,4]]
Explanation:
1 + 2 + 6 = 9
1 + 3 + 5 = 9
2 + 3 + 4 = 9
There are no other valid combinations.
Example 3:

Input: k = 4, n = 1
Output: []
Explanation: There are no valid combinations.
Using 4 different numbers in the range [1,9], the smallest sum we can get is 1+2+3+4 = 10 and since 10 > 1, there are no valid combination.
```
```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        res = []
        path = []
        sum_now = 0
        def backtrack(k, n, startindex, sum_now):
            if sum_now > n:
                return 
            if len(path) == k:
                if sum_now == n:
                    res.append(path[:])
                return 
            for i in range(startindex, 10 - (k-len(path)) + 1):
                path.append(i)
                sum_now += i
                backtrack(k, n, i+1, sum_now) # details 
                path.pop()
                sum_now -= i  
        backtrack(k, n, 1, 0)
        return res 
``` 


### 377. Combination Sum IV 
Given an array of distinct integers nums and a target integer target, return the number of possible combinations that add up to target.

The test cases are generated so that the answer can fit in a 32-bit integer.
```
Example 1:

Input: nums = [1,2,3], target = 4
Output: 7
Explanation:
The possible combination ways are:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
Note that different sequences are counted as different combinations.
Example 2:

Input: nums = [9], target = 3
Output: 0
```
> DP 背包问题？
```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        # 完全背包，但是遍历顺序不一样
        dp = [0] * (target + 1)
        dp[0] = 1
        for i in range(1, target+1):
            for j in nums:
                if i >= j:
                    dp[i] += dp[i - j]
        return dp[-1]
``` 


### 254. Factor Combinations 
Numbers can be regarded as the product of their factors.

For example, 8 = 2 x 2 x 2 = 2 x 4.
Given an integer n, return all possible combinations of its factors. You may return the answer in any order.

Note that the factors should be in the range [2, n - 1].
```
Example 1:

Input: n = 1
Output: []
Example 2:

Input: n = 12
Output: [[2,6],[3,4],[2,2,3]]
Example 3:

Input: n = 37
Output: []
```

```python
class Solution:
    #Iterative:
    def getFactors(self, n):
        todo, combis = [(n, 2, [])], []
        while todo:
            n, i, combi = todo.pop()
            while i * i <= n:
                if n % i == 0:
                    combis += combi + [i, n/i],
                    todo += (n/i, i, combi+[i]),
                i += 1
        return combis

    #Recursive:
    def getFactors(self, n):
        def factor(n, i, combi, combis):
            while i * i <= n:
                if n % i == 0:
                    combis += combi + [i, n/i],
                    factor(n/i, i, combi+[i], combis)
                i += 1
            return combis
        return factor(n, 2, [], [])
``` 



### 17. Letter Combinations of a Phone Number 
Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.
```
Example 1:

Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
Example 2:

Input: digits = ""
Output: []
Example 3:

Input: digits = "2"
Output: ["a","b","c"]
```

```python
# class Solution:
#     def letterCombinations(self, digits: str) -> List[str]:
        
class Solution:
    def __init__(self):
        self.answers: List[str] = []
        self.answer: str = ''
        self.letter_map = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }
    def letterCombinations(self, digits: str) -> List[str]:
        self.answers.clear()
        if not digits: return []
        self.backtracking(digits, 0)
        return self.answers
    
    def backtracking(self, digits: str, index: int) -> None:
        # 回溯函数没有返回值
        # Base Case
        if index == len(digits):    # 当遍历穷尽后的下一层时
            self.answers.append(self.answer)
            return 
        # 单层递归逻辑  
        # letters: str = self.letter_map[digits[index]]
        letters = self.letter_map[digits[index]]
        for letter in letters:
            self.answer += letter   # 处理
            self.backtracking(digits, index + 1)    # 递归至下一层
            self.answer = self.answer[:-1]  # 回溯
``` 

### 46. Permutations
Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.
```
Example 1:

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
Example 2:

Input: nums = [0,1]
Output: [[0,1],[1,0]]
Example 3:

Input: nums = [1]
Output: [[1]]
``` 

```python
class Solution:
    def permute(self, nums):
        res = []
        path = []
        def dfs(nums):
            if len(nums) == len(path):
                res.append(path[:])
                return 
            for i in range(len(nums)):
                if nums[i] in path: # 这个是关键，去重，否则很多一样的
                    continue
                path.append(nums[i])
                dfs(nums)
                path.pop()
        dfs(nums)
        return res
``` 


### 47. Permutations II
Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.
```
Example 1:

Input: nums = [1,1,2]
Output:
[[1,1,2],
 [1,2,1],
 [2,1,1]]
Example 2:

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```
```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        # res用来存放结果
        if not nums: return []
        res = []
        used = [0] * len(nums)
        def backtracking(nums, used, path):
            # 终止条件
            if len(path) == len(nums):
                res.append(path.copy())
                return
            for i in range(len(nums)):
                if not used[i]:
                    if i>0 and nums[i] == nums[i-1] and not used[i-1]:
                        continue
                    used[i] = 1
                    path.append(nums[i])
                    backtracking(nums, used, path)
                    path.pop()
                    used[i] = 0
        # 记得给nums排序
        backtracking(sorted(nums),used,[])
        return res
``` 


### 77. Combinations 
Given two integers n and k, return all possible combinations of k numbers out of the range [1, n].

You may return the answer in any order.
```
Example 1:

Input: n = 4, k = 2
Output:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
Example 2:

Input: n = 1, k = 1
Output: [[1]]
```
```python
class Solution:
    def combine(self, n,k):
        res = []
        path = []
        def dfs(n,k, index):
            if len(path) == k:
                res.append(path[:])
                return 
            for i in range(index, n + 1):
                path.append(i)
                dfs(n,k, i + 1)
                path.pop()

        dfs(n,k,1)
        return res
    
# class Solution:
#     def combine(self, n: int, k: int) -> List[List[int]]:  
#         # # 剪枝 - 效果还挺明显的！
#         res=[]  #存放符合条件结果的集合
#         path=[]  #用来存放符合条件结果
#         def backtrack(n,k,startIndex):
#             if len(path) == k:
#                 res.append(path[:])
#                 return 
#             for i in range(startIndex,n - (k - len(path)) + 2):  #优化的地方
#                 path.append(i)  #处理节点 
#                 backtrack(n,k,i+1)  #递归
#                 path.pop()  #回溯，撤销处理的节点
#         backtrack(n,k,1)
#         return res
    
    
          # 代码回想录模板  
#         res = []
#         path = []
#         def backtrack(n, k, StartIndex):
#             if len(path) == k:
#                 res.append(path[:])
#                 return
#             for i in range(StartIndex, n + 1):
#                 path.append(i)
#                 backtrack(n, k, i+1)
#                 path.pop()
#         backtrack(n, k, 1)
#         return res
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
