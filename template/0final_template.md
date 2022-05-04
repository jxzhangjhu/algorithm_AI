All topics about coding test including ML coding

- Double pointer & sliding window 
- Recursion & backtracking - done 
- Graph DFS
- Graph BFS
- Tree
- Stack
- Heap (Priority Queue) - done
- DP - done 
- Linked list 
- Binary search 
- Array/String 
- Sorting 
- Prefix sum 
- Math/random/statistics/probability
- ML coding 





# Double pointer & sliding window 


# Recursion & backtracking 
很重要明确几个步骤
1. 确定递归函数的参数和返回值
2. 确定终止条件 
3. 确定递归回溯的遍历过程，一般是for loop，横向遍历，递归是纵向遍历

主要解决的问题
1. 组合问题：N个数里面按一定规则找出k个数的集合
2. 排列问题：N个数按一定规则全排列，有几种排列方式
3. 切割问题：一个字符串按一定规则有几种切割方式
4. 子集问题：一个N个数的集合里有多少符合条件的子集
5. 棋盘问题：N皇后，解数独等等

## tempalte 
```python

def dfs(参数):
    if (终止条件):
        存放结果
        return 

    for (选择本层集合中的元素，一般是集合大小)
        处理当前节点
        dfs(路径，选择列表)
        回溯，撤销处理结果 pop操作
```
这是fuxuemingzhu的模板 关于subset的几个题
```python
res = []
path = []

def backtrack(未探索区域, res, path):
    if path 满足条件:
        res.add(path) # 深度拷贝
        # return  # 如果不用继续搜索需要 return
    for 选择 in 未探索区域当前可能的选择:
        if 当前选择符合要求:
            path.add(当前选择)
            backtrack(新的未探索区域, res, path)
            path.pop()

# 作者：fuxuemingzhu
# 链接：https://leetcode-cn.com/problems/subsets-ii/solution/hui-su-fa-mo-ban-tao-lu-jian-hua-xie-fa-y4evs/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```
> 关于deep copy的问题
``` python
s = []
a = [1,2]
s.append(a)
print(s)
a.pop()
a.append(3)
s.append(a)
print(s)
# [[1, 2]]
# [[1, 3], [1, 3]]

s = []
a = [1,2]
s.append(a[:])
print(s)
a.pop()
a.append(3)
s.append(a)
print(s)
# [[1, 2]]
# [[1, 2], [1, 3]]
```


## Examples 
22. Generate Parentheses https://leetcode.com/problems/generate-parentheses/ 
```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        ans = []
        def backtrack(S = [], left = 0, right = 0):
            if len(S) == 2 * n:
                ans.append("".join(S))
                return
            if left < n:
                S.append("(")
                backtrack(S, left+1, right)
                S.pop()
            if right < left:
                S.append(")")
                backtrack(S, left, right+1)
                S.pop()
        backtrack()
        return ans
    
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        if n <= 0: return 
        res = []
        def dfs(paths, num_left, num_right):
            if num_left > n or num_left < num_right: # 这个有讲究，关键是括号的处理
                return 
            # 先写递归出口
            if len(paths) == n * 2:
                res.append(paths)
                return 
            # 递归定义， 只有paths 放进去作为参数
            dfs(paths + '(', num_left + 1, num_right)
            dfs(paths + ')', num_left, num_right + 1)
        dfs('', 0, 0)
        return res 
```

17. Letter Combinations of a Phone Number https://leetcode.com/problems/letter-combinations-of-a-phone-number/
```python
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

77. Combinations https://leetcode.com/problems/combinations/ 

```python
class Solution:
    def combine(self, n,k):
        res = []
        path = []
        def dfs(n,k, index):
            if len(path) == k:
                res.append(path[:]) # 这个地方关键是必须用path[:]这个含义是copy path的内容， 新内容，如果知识path还是之前的id
                return 
            for i in range(index, n + 1):
                path.append(i)
                dfs(n,k, i + 1)
                path.pop()

        dfs(n,k,1)
        return res

# 优化
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res=[]  #存放符合条件结果的集合
        path=[]  #用来存放符合条件结果
        def backtrack(n,k,startIndex):
            if len(path) == k:
                res.append(path[:])
                return
            for i in range(startIndex,n-(k-len(path))+2):  #优化的地方
                path.append(i)  #处理节点 
                backtrack(n,k,i+1)  #递归
                path.pop()  #回溯，撤销处理的节点
        backtrack(n,k,1)
        return res
```


78. Subsets https://leetcode.com/problems/subsets/ 

Given an integer array nums of unique elements, return all possible subsets (the power set). The solution set must not contain duplicate subsets. Return the solution in any order. 

Examples 1: 
```
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

```python
class Solution:
    def subsets(self, nums):
        res = []
        path = []
        def dfs(nums, index):
            res.append(path[:]) # 等价于deepcopy 直接copy里面的内容，这里面path没有终止条件
            # res.append(copy.deepcopy(path))
            # stop critiera
            if len(nums) == index: 
                return 
            for i in range(index, len(nums)):
                path.append(nums[i])
                dfs(nums, i + 1) # 这个地方不是index + 1 而是i + 1
                path.pop()
        dfs(nums, 0)
        return res 
```

90. Subsets II https://leetcode.com/problems/subsets-ii/ 

Given an integer array nums that may contain duplicates, return all possible subsets (the power set). The solution set must not contain duplicate subsets. Return the solution in any order.

Examples 1: 
```
Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
```
```python 
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []
        def dfs(nums, index):
            res.append(path[:]) # 等价于deepcopy 直接copy里面的内容，这里面path没有终止条件
            # res.append(copy.deepcopy(path))
            # stop critiera
            if len(nums) == index: 
                return 
            for i in range(index, len(nums)):
                if i > index and nums[i] == nums[i-1]: # 加这个判定条件！ # 当前后元素值相同时，跳入下一个循环，去重 
                    continue 
                path.append(nums[i])
                dfs(nums, i + 1) # 这个地方不是index + 1 而是i + 1
                path.pop()
        dfs(nums, 0)
        return res  
```


46. Permutations  https://leetcode.com/problems/permutations/ 
> Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order. 

```
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
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

47. Permutations II https://leetcode.com/problems/permutations-ii/ 
> Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order. 
Example 1:
```
Input: nums = [1,1,2]
Output:
[[1,1,2],
 [1,2,1],
 [2,1,1]]
```
Example 2:
```
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




# Graph (DFS/BFS)

733. Flood Fill  https://leetcode.com/problems/flood-fill/ 
> BFS or DFS 都可以，但似乎DFS更快！BFS没有超时，需要加一个判断，这个和number of island不同在于，不需要记录visit，直接全部赋值？

```python 
class Solution:
    def floodFill(self, image, sr, sc, newColor):
        curr_color = image[sr][sc]
        if curr_color == newColor: # 如果不加这个就超时
            return image

        n, m = len(image), len(image[0])
        direct = [(0, 1),(0, -1),(1, 0),(-1, 0)]
        from collections import deque
        queue = collections.deque([]) # 先加[] 
        queue.append((sr, sc))
        image[sr][sc] = newColor
        # visit = set() # 这个题不适合加visit，why？ 没有必要？BFS大部分都需要加
        while queue:
            curr_x, curr_y = queue.popleft()
            for dx, dy in direct:
                x = curr_x + dx
                y = curr_y + dy
                if 0 <= x < n and 0 <= y < m and image[x][y] == curr_color:
                # if 0 <= x < m and 0 <= y < n and image[x][y] == curr_color and (x, y) in visit: 加了这个就说out of range, 不明白！
                    visit.add((x,y))
                    queue.append((x,y))
                    image[x][y] = newColor
                # else:
                #     continue
        return image
```


200. Number of Islands  https://leetcode.com/problems/number-of-islands/ 






# Tree


# Stack

# Heap (Priority Queue)
> python 官方文档 https://docs.python.org/zh-cn/3/library/heapq.html 
```python
import heapq

s = [4,2,1,4,6]
heapq.heapify(s) # s已经堆化
# print(s) #[1, 2, 4, 4, 6], 只能保证最小的在最上面，后面无法保证

s1 = [4,3,2,1]
heapq.heappush(s1,5)
heapq.heappush(s1,3)
heapq.heappush(s1,4)
print(s1) #[4, 3, 2, 1, 5, 3, 4] 
heapq.heappop(s1) #4 

# 说明，s1已经被创建了，heappush并不能保证最小的在堆顶，只是正常的pop
s1 = [4,3,2,1]
heapq.heapify(s1) # 对s1堆化就可以了
heapq.heappush(s1,5)
heapq.heappush(s1,3)
print(s1)# [1, 3, 2, 4, 5, 3]
heapq.heappop(s1) #1 
heapq.heappop(s1) #2
heapq.heappop(s1) #2
heapq.heappop(s1) #3 



s1 = []
heapq.heappush(s1,4)
heapq.heappush(s1,2)
heapq.heappush(s1,3)
heapq.heappush(s1,1)
heapq.heappush(s1,2)
heapq.heappush(s1,5)
print(s1) #[1, 2, 3, 4, 2, 5] 
heapq.heappop(s1) #1 

```


## 定义
1. 分为最小堆minheap，和最大堆，maxheap， 也就是最小元素或者最大元素在堆顶
2. 堆是一个完全二叉树，但堆的底层实现一般是数组，而不是二叉树
3. 孩子节点都比父亲节点大， 但是左右孩子的大小不影响
4. 堆不是binary search tree 
5. 堆的操作是从上到下， 从左到右

## 基本操作 - 高度是logn 
1. 构建堆 heapify - o(n)
2. 遍历堆 o(nlogn)
3. add - o(n) 
4. remove - 理论上是o(logn) 但实际上python的库函数是for loop遍历的，所以是o(n)
5. pop - o(logn)， push 也是o(logn)
6. min or max - o(1) 
7. 由于是数组操作，选定k，父亲是k/2，左孩子kx2, 右孩子kx2+1
8. 可以结合hashmap 去查询或者remove 指定值 

## 使用条件
1. 找最大值或者最小值（60%）
2. 找第k大（pop k 次 复杂度o(nlogk)） (50%)
3. 要求logn时间对数据进行操作 （40%）  

##堆不能解决的问题
1. 查询比某个数大的最小值或者最近接的值 (平衡二叉树 balanced bst 才可以解决)
2. 找某段区间的最大值最小值 （线段树segmenttree 可以解决）
3. o(n) 找第k大的数 （需要使用快排中的partition操作）

## template 
```python

from heapq import heappush, heappop

class Heap:
    def __init__(self):
        self.minheap()
        self.deleted_set = set()

    def push(self, index, val):
        heappush(self.minheal, (val, index))

    def _lazy_deletion(self):
        while self.minheap and self.minheap[0][1] in self.deleted_set:
            heappop(self.minheap)

    def top(self):
        self._lazy_deletion()
        return self.minheap[0]

    def pop(self): # 移除顶端元素
        self._lazy_deletion()
        heappop(self.minheap)

    def delete(self, index):
        self.deleted_set.add(index)

    def is_empty(self):
        return not bool(self.minheap)

```

### Example 264. Ugly Number II 
https://leetcode.com/problems/ugly-number-ii/ 

An ugly number is a positive integer whose prime factors are limited to 2, 3, and 5.

Given an integer n, return the nth ugly number.
 
Example 1:
```
Input: n = 10
Output: 12
Explanation: [1, 2, 3, 4, 5, 6, 8, 9, 10, 12] is the sequence of the first 10 ugly numbers.
```
Example 2:
```
Input: n = 1
Output: 1
Explanation: 1 has no prime factors, therefore all of its prime factors are limited to 2, 3, and 5.
```
>solution 
```python
        # 用heap来做，不断的找最小值，然后push回去, 时间复杂度 o(nlogn)
    
        import heapq
        
        heap = [1]
        visited = set([1])
        
        min_val = None
        for i in range(n): # n 次操作，也是nth的最小值
            min_val = heapq.heappop(heap)
            for factor in [2, 3, 5]:
                if min_val * factor not in visited:
                    visited.add(min_val * factor)
                    heapq.heappush(heap, min_val * factor)
                    
        return min_val
```

### 973. K Closest Points to Origin
https://leetcode.com/problems/k-closest-points-to-origin/

Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k closest points to the origin (0, 0).

The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)2 + (y1 - y2)2).

You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).

Example 1:
```
Input: points = [[1,3],[-2,2]], k = 1
Output: [[-2,2]]
Explanation:
The distance between (1, 3) and the origin is sqrt(10).
The distance between (-2, 2) and the origin is sqrt(8).
Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
We only want the closest k = 1 points from the origin, so the answer is just [[-2,2]].
```
Example 2:
```
Input: points = [[3,3],[5,-1],[-2,4]], k = 2
Output: [[3,3],[-2,4]]
Explanation: The answer [[-2,4],[3,3]] would also be accepted.
```

>solution

```python
#         # heap  最小堆 - 把所有点都放入最小堆，然后用最小堆取出k个，时间o(nlogn) 空间o(n) + o(k)
#         # 遍历堆 o(nlogn) 所以时间是这个level， 空间是因为开了heap + res的部分，单独储存res
        
        
#         import heapq
#         heap = []
        
#         for point in points:
#             cur_dis = point[0] ** 2 + point[1] ** 2
#             heapq.heappush(heap, (cur_dis, point)) # 注意加进去的时候还是看cur_dis 自动排序好了
            
#         res = []
#         i = 0 
#         while i < k: 
#             _, point = heapq.heappop(heap)
#             res.append(point)
#             i += 1
        
        # return res
    
        # 最大堆 更优一些， 因为不需要重新开整个heap - 时间 o(nlogk), 空间o(k) 
        heap = []
        for point in points:
            cur_dis = point[0] ** 2 + point[1] ** 2
            heapq.heappush(heap, (-cur_dis, point)) # 注意加进去的时候还是看cur_dis 自动排序好了
            if len(heap) > k:
                heapq.heappop(heap)
        
        res = []
        i = 0 
        while i < k: 
            _, point = heapq.heappop(heap)
            res.append(point)
            i += 1
        
        return res
```

### Example: Lintcode 545 · Top k Largest Numbers II
https://www.lintcode.com/problem/545/ 

Description
Implement a data structure, provide two interfaces:

add(number). Add a new number in the data structure.
topk(). Return the top k largest numbers in this data structure. k is given when we create the data structure.

>solution
```python
import heapq
class Solution:
    """
    @param: k: An integer
    """
    def __init__(self, k):
        # do intialization if necessary
        self.k = k
        self.heap = []

    """
    @param: num: Number to be added
    @return: nothing
    """
    def add(self, num):
        # write your code here
        heapq.heappush(self.heap, num)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)


    """
    @return: Top k element
    """
    def topk(self):
        # write your code here
        return sorted(self.heap, reverse=True)
```

### 253. Meeting Rooms II
https://leetcode.com/problems/meeting-rooms-ii/

Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required.

Example 1:
```
Input: intervals = [[0,30],[5,10],[15,20]]
Output: 2
```
Example 2:
```
Input: intervals = [[7,10],[2,4]]
Output: 1
```

>solution
https://leetcode.com/problems/meeting-rooms-ii/solution/ 

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        
        # If there is no meeting to schedule then no room needs to be allocated.
        if not intervals:
            return 0

        # The heap initialization
        free_rooms = []

        # Sort the meetings in increasing order of their start time.
        intervals.sort(key= lambda x: x[0])

        # Add the first meeting. We have to give a new room to the first meeting.
        heapq.heappush(free_rooms, intervals[0][1])

        # For all the remaining meeting rooms
        for i in intervals[1:]:

            # If the room due to free up the earliest is free, assign that room to this meeting.
            if free_rooms[0] <= i[0]:
                heapq.heappop(free_rooms)

            # If a new room is to be assigned, then also we add to the heap,
            # If an old room is allocated, then also we have to add to the heap with updated end time.
            heapq.heappush(free_rooms, i[1])

        # The size of the heap tells us the minimum rooms required for all the meetings.
        return len(free_rooms)

# Complexity Analysis

# Time Complexity: O(NlogN)

# There are two major portions that take up time here. One is sorting of the array that takes O(NlogN) considering that the array consists of NN elements.Then we have the min-heap. In the worst case, all NN meetings will collide with each other. In any case we have NN add operations on the heap. In the worst case we will have NN extract-min operations as well. Overall complexity being (NlogN)(NlogN) since extract-min operation on a heap takes O(logN).

# Space Complexity: O(N) because we construct the min-heap and that can contain NN elements in the worst case as described above in the time complexity section. Hence, the space complexity is O(N).
```


### Example： 373. Find K Pairs with Smallest Sums
https://leetcode.com/problems/find-k-pairs-with-smallest-sums/ 

You are given two integer arrays nums1 and nums2 sorted in ascending order and an integer k.

Define a pair (u, v) which consists of one element from the first array and one element from the second array.

Return the k pairs (u1, v1), (u2, v2), ..., (uk, vk) with the smallest sums.

Example 1:
```
Input: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
Output: [[1,2],[1,4],[1,6]]
Explanation: The first 3 pairs are returned from the sequence: [1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]
```
Example 2:
```
Input: nums1 = [1,1,2], nums2 = [1,2,3], k = 2
Output: [[1,1],[1,1]]
Explanation: The first 2 pairs are returned from the sequence: [1,1],[1,1],[1,2],[2,1],[1,2],[2,2],[1,3],[1,3],[2,3]
```
Example 3:
```
Input: nums1 = [1,2], nums2 = [3], k = 3
Output: [[1,3],[2,3]]
Explanation: All possible pairs are returned from the sequence: [1,3],[2,3]
```

>solution 
```python
# 用heap来解  - 和973 很像，但是是二维遍历，然后也是用最大堆
        import heapq
        heap = []
        for i in range(min(k, len(nums1))):
            for j in range(min(k, len(nums2))): 
                if len(heap) < k:
                    heapq.heappush(heap, ((-nums1[i] - nums2[j]), i, j))
                else: 
                    if nums1[i] + nums2[j] < -heap[0][0]:
                        heappop(heap)
                        heappush(heap, (-(nums1[i] + nums2[j]), i, j))
        res = []
        for _,i,j in heap:
            res.append( [nums1[i], nums2[j]])  
        
        return res 
    
        # time - klogk? 不太确定
        # space - o(k)
```

### Example: 215. Kth Largest Element in an Array
https://leetcode.com/problems/kth-largest-element-in-an-array/
Given an integer array nums and an integer k, return the kth largest element in the array.

Note that it is the kth largest element in the sorted order, not the kth distinct element.

Example 1:
```
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
```
Example 2:
```
Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4
```

>solution
```python
        # 暴力干 - 但没有什么意思， python 的sort 时间复杂度是多少？ 不清楚 
        # nums.sort()
        # return nums[-k]
    
# That would be an algorithm of O(NlogN) time complexity and 
# O(1) space complexity.
    
        # heap 
        import heapq 
        return heapq.nlargest(k, nums)[-1]
    
# Time complexity : O(Nlogk).
# Space complexity : O(k) to store the heap elements.
```


### Example : 692. Top K Frequent Words 
https://leetcode.com/problems/top-k-frequent-words/ 

Given an array of strings words and an integer k, return the k most frequent strings.

Return the answer sorted by the frequency from highest to lowest. Sort the words with the same frequency by their lexicographical order.

 
Example 1:
```
Input: words = ["i","love","leetcode","i","love","coding"], k = 2
Output: ["i","love"]
Explanation: "i" and "love" are the two most frequent words.
Note that "i" comes before "love" due to a lower alphabetical order.
```
Example 2:
```
Input: words = ["the","day","is","sunny","the","the","the","sunny","is","is"], k = 4
Output: ["the","is","sunny","day"]
Explanation: "the", "is", "sunny" and "day" are the four most frequent words, with the number of occurrence being 4, 3, 2 and 1 respectively.
```

>Solution
```python
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        
        # heap 很显然的解法，但不想用这么多额外的开销
        counts = defaultdict(lambda: 0)
        for word in words:
            counts[word] += 1
            
        inverse = defaultdict(lambda: [])
        for word, count in counts.items():
            heappush(inverse[count], word)
        
        res = []
        
        for count in nlargest(k, counts.values()):
            res.append(heappop(inverse[count]))
            
        return res     
```


### Example: 658. Find K Closest Elements
https://leetcode.com/problems/find-k-closest-elements/ 

Given a sorted integer array arr, two integers k and x, return the k closest integers to x in the array. The result should also be sorted in ascending order.

An integer a is closer to x than an integer b if:
```
|a - x| < |b - x|, or
|a - x| == |b - x| and a < b
 ```

Example 1:
```
Input: arr = [1,2,3,4,5], k = 4, x = 3
Output: [1,2,3,4]
```
Example 2:
```
Input: arr = [1,2,3,4,5], k = 4, x = -1
Output: [1,2,3,4]
```

>Solution 
```python
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        # way 1 - double pointer 
        # way 2 - binary search 
        
        # way 3 - heap 
        import heapq 
        heap = [] 
        
        for num in arr:
            dis = abs(num - x)
            heapq.heappush(heap, (dis, num))
            # if len(heap) > k: # 用最大堆有问题，就是会忽视这个条件 |a - x| == |b - x| and a < b
            #     heapq.heappop(heap)
            
        res = []
        i = 0
        while i < k:
            _, num = heapq.heappop(heap)
            res.append(num)
            i += 1
            
        return sorted(res) # 别忘了最后要sorted()  如果是res.sort() 会返回[]

```



# DP
## 问题特征 
1. 求最大或者最小, 不要求返回哪一个
2. 如果求所有可能一般都需要backtracking

## 问题类型 - 1D，2D
1. 区间DP
2. 背包DP

## 求解思路
1. 确定DP数组以及下标的含义
2. 确定递推公式
3. 初始化DP数组
4. 确定遍历顺序
5. 举例推导dp数组

## Examples 
53. Maximum Subarray 
https://leetcode.com/problems/maximum-subarray/
> 1D DP数组，求最大值，这里面要注意判断dp[i-1]的正负，不是nums[i]的正负，同时注意subarry是连续的！

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # dp[i] 数组的含义是以nums[i]结尾的连续子数组的最大和
        # 关键是dp[i] 转换的含义，这里面要判断dp[i-1]的大小，如果>0, dp[i]会更大，但是dp[i-1]<0, 重新开始，因为nums[i]会更小如果加上dp[i-1] 
    
        n = len(nums)
        dp = [0] * n 
        dp[0] = nums[0]
        for i in range(1, n):
            if dp[i-1] > 0: #不是判断nums[i]的正负
                dp[i] = dp[i-1] + nums[i]
            else:
                dp[i] = nums[i]
        return max(dp) # 返回所有的最大值
        
        # brute force - 这是比较优化的方法，但超时
        res = -inf 
        n = len(nums)
        for i in range(n):
            curr = 0 
            for j in range(i, n):
                curr += nums[j]
                res = max(res, curr)
        
        return res
```

152. Maximum Product Subarray https://leetcode.com/problems/maximum-product-subarray/ 
> 和53非常像，但是是求和，需要维护max和min同时，现在状态并不是由上一个状态完全决定 - maximum, fits DP problem
``` python
        # dp 解法和之前不一样的
        n = len(nums)
        dpmax = [0] * n
        dpmin = [0] * n
        dpmax[0] = nums[0]
        dpmin[0] = nums[0]
        res = nums[0]
        
        for i in range(1, n):
            dpmax[i] = max(dpmax[i-1] * nums[i], nums[i], dpmin[i-1]*nums[i])
            dpmin[i] = min(dpmax[i-1] * nums[i], nums[i], dpmin[i-1]*nums[i])
            res = max(res, dpmax[i])
        
        return res
```

926. Flip String to Monotone Increasing 
https://leetcode.com/problems/flip-string-to-monotone-increasing/ 

> [2D DP数组类型] 求最小，符合DP的特征，同时2种情况讨论，确定用2d dp，关键在于如何确定dp数组的含义！
```python
class Solution: # dp 看题解思路 O(n) O(n)
    def minFlipsMonoIncr(self, S: str) -> int:
        # DP数组的含义才是最重要的
        # DP[i][0] 是到i这个位置的子串，结尾是0的，需要的最少次数
        # DP[i][1] 是i这个位置的子串，结尾是1的，需要的最少次数
        # 递归过程：s[i] = 0时有2种情况
        # 1. 对于dp[i][0]需要都是0, 所以不需要翻转dp[i][0] = dp[i-1][0]
        # 2. 对于dp[i][1]需要翻转，因此+1，这个时候是上个时刻最小值+1 dp[i][1] = min(dp[i-1][0], dp[i-1][1]) + 1
        # 当s[i] = 1时也有2种情况
        # 1. dp[i][0] 需要翻转 dp[i][0] = dp[i-1][0] + 1 也就是从1到0
        # 2. dp[i][1] 不需要翻转 dp[i][1] = min(dp[i - 1][0], dp[i - 1][1]) 都是0的时候翻转 
        # 特殊情况
        if not S: return 0
        n = len(S)
        # 递归初始化
        dp = [[0] * 2 for _ in range(n)]
        dp[0][0] = int(S[0] == '1')
        dp[0][1] = int(S[0] == '0')
        # 递归遍历从1到n
        for i in range(1, n):
            if S[i] == '0':
                dp[i][0] = dp[i - 1][0]
                dp[i][1] = min(dp[i - 1][0], dp[i - 1][1]) + 1
            else:
                dp[i][0] = dp[i - 1][0] + 1
                dp[i][1] = min(dp[i - 1][0], dp[i - 1][1])
        return min(dp[n - 1]) # 返回值，最大到n-1

class Solution: # dp 空间优化 O(n) O(1)
    def minFlipsMonoIncr(self, S: str) -> int:
        zero, one = 0, 0
        n = len(S)
        for i in range(n):
            if S[i] == '0':
                one = min(zero, one) + 1
            else:
                zero, one = zero + 1, min(zero, one)
        return min(one, zero)

class Solution: # 前缀和 看题解思路 O(n) O(n)
    def minFlipsMonoIncr(self, S: str) -> int:
        P = [0]
        n = len(S)
        for a in S:
            P.append(P[-1] + int(a))
        # return min(P[i] + n - i - (P[n] - P[i]) for i in range(n))
        return min(P[i] + n - i - (P[n] - P[i]) for i in range(n + 1))
        # return min(P[j] + len(S)-j-(P[-1]-P[j]) for j in range(len(P)))
```


416. Partition Equal Subset Sum https://leetcode.com/problems/partition-equal-subset-sum/ 

>  好题 + 这个帖子好，顺便复习0-1背包，完全背包
https://leetcode-cn.com/problems/partition-equal-subset-sum/solution/by-flix-szk7/
https://leetcode-cn.com/problems/partition-equal-subset-sum/solution/yi-pian-wen-zhang-chi-tou-bei-bao-wen-ti-a7dd/ 

01 背包问题的定义
![p](https://github.com/jxzhangjhu/Coding2022/blob/main/images/beibao.png)


```python

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        # brute force - DFS 的方法
        # DP  背包模板转换 
        # https://leetcode-cn.com/problems/partition-equal-subset-sum/solution/by-flix-szk7/ 
        
        # way 1 - 2D DP 
        # dp[i][j] 前i个数字选取若干个，刚好选出的数字和为j
        total = sum(nums)
        if total % 2 == 1: return False 
        
        target = total // 2
        if max(nums) > target: return False
        
        n = len(nums) # 注意初始化要target + 1, and n+1 最终状态dp[n][target]
        dp = [[False] * (target+1) for _ in range(n + 1)]
        dp[0][0] = True
        for i in range(1, n + 1):
            for j in range(target + 1):
                if j < nums[i-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] | dp[i-1][j - nums[i-1]] # | 运算
        return dp[n][target]
    
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        # 套用背包的模板，第一遍确实想不清楚
        taraget = sum(nums)
        if taraget % 2 == 1: return False
        taraget //= 2
        dp = [0] * 10001
        for i in range(len(nums)):
            for j in range(taraget, nums[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
        return taraget == dp[taraget]
```

<!-- 其他还有好几个背包问题可以一起做一下 -->

322. Coin Change https://leetcode.com/problems/coin-change/ 
> 也是背包问题一起解决一下
```python
         # 记整数数组 coins 的长度为 nn。为便于状态更新，减少对边界的判断，初始二维 dpdp 数组维度为 {(n+1) \times (*)}(n+1)×(∗)，其中第一维为 n+1n+1 也意味着：第 ii 种硬币为 coins[i-1]coins[i−1]，第 11 种硬币为 coins[0]coins[0]，第 00 种硬币为空。

        n = len(coins)
        dp = [[amount+1] * (amount+1) for _ in range(n+1)]    # 初始化为一个较大的值，如 +inf 或 amount+1
        # 合法的初始化
        dp[0][0] = 0    # 其他 dp[0][j]均不合法
        
        # 完全背包：优化后的状态转移
        for i in range(1, n+1):             # 第一层循环：遍历硬币
            for j in range(amount+1):       # 第二层循环：遍历背包
                if j < coins[i-1]:          # 容量有限，无法选择第i种硬币
                    dp[i][j] = dp[i-1][j]
                else:                       # 可选择第i种硬币
                    dp[i][j] = min( dp[i-1][j], dp[i][j-coins[i-1]] + 1 )

        ans = dp[n][amount] 
        return ans if ans != amount+1 else -1
```

198. House Robber https://leetcode.com/problems/house-robber/
> 经典dp，注意初始条件，边界条件，特殊判定
```python
        # dp[i] 定义 - 前i个房子最大值
        # dp转移 dp[i] = max(dp[i-1], dp[i-2] + nums[i]])
        # 初始化 dp
        # 这些特判也很重要！
        if len(nums) == 0: 
            return 0
        if len(nums) == 1:
            return nums[0]
        n = len(nums)
        dp = [0] * n 
        dp[0] = nums[0] # 第一个简单
        dp[1] = max(nums[0], nums[1]) # 第二个是前两个最大值
        for i in range(2, n):
            dp[i] = max(dp[i-1], dp[i-2] + nums[i]) # 注意nums[i] 所以只能是到n
        return dp[-1] # 返回最后一个数比dp[n-1]安全？
```


62. Unique Paths https://leetcode.com/problems/unique-paths/ 
> 2D矩阵的跳楼梯 - dp[i][j] 含义是到[i,j]这个状态的所有数目，只能从2个状态得来！dp[i-1][j] and dp[i][j-1]. 

DP问题的一个核心思想是
1. 寻找子问题，缩小问题规模
2. 枚举子问题到原问题的所有转移过程和状态，枚举！
3. 明确dp数组的含义，一般直接针对问题本身，max，min，all possible， 最值问题，存在问题，组合问题
4. 初始化的定义，这个需要经验，看corn case
5. 明确遍历顺序和最后输出，是最终状态，还是所有的最值

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # 直接开一个2维数组
        dp = [[1 for i in range(n)] for j in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i][j - 1] + dp[i - 1][j] # 递推公式
        return dp[m - 1][n - 1]
```

55. Jump Game https://leetcode.com/problems/jump-game/ 
> 更像是greedy 贪心策略，DP没有那么直接. 几个思路，仔细想想，其实不难

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        """
        :type nums: List[int]
        :rtype: bool
        """
        length = len(nums)
        dp = [0] * length
        dp[0] = nums[0]
        for i in range(1, length - 1):            
            if dp[i - 1] < i:
                return False
            dp[i] = max(i + nums[i], dp[i - 1])
            if dp[i] >= length - 1:
                return True
        return dp[length - 2] >= length - 1       
        
    
        # # 改写45 - 贪心
        # maxstep = 0
        # n = len(nums)
        # for i in range(n):
        #     if i <= maxstep:
        #         maxstep = max(maxstep, nums[i] + i)
        #         if maxstep >= n - 1:
        #             return True
        # return False
        
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        current_max = 0
        
        for i in range(len(nums)):
            if current_max < i: return False
            current_max = max(current_max, i + nums[i])
            if current_max >= len(nums)-1: return True
            
        return current_max >= len(nums)-1
```

377. Combination Sum IV https://leetcode.com/problems/combination-sum-iv/
> 完全背包问题 dp[i]含义是

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

221. Maximal Square https://leetcode.com/problems/maximal-square/ 
> 这个状态方程根本想不到

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return 0
        
        maxSide = 0
        rows, columns = len(matrix), len(matrix[0])
        dp = [[0] * columns for _ in range(rows)]
        for i in range(rows):
            for j in range(columns):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1 #核心在这
                    maxSide = max(maxSide, dp[i][j])
        
        maxSquare = maxSide * maxSide
        return maxSquare
```

300. Longest Increasing Subsequence  https://leetcode.com/problems/longest-increasing-subsequence/ 
> 可以理解，但是自己想是没有想到 https://leetcode-cn.com/problems/longest-increasing-subsequence/solution/zui-chang-shang-sheng-zi-xu-lie-dong-tai-gui-hua-e/ 
```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # 非常类似wordbreak 定义写对了，但是转移方程还是犹豫了！
        # dp[i] 定义为前i个的最长上升数
        # dp[i] 和 dp[i-1] 比还是跟之前所有的比, 确实比较像！
        dp = [1] * (len(nums))
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
```

# Linked list 


# Binary search 


# Array/String 

# Sorting 

# Prefix sum 





<!-- 






# Binary search 搜索 

### 隐式二分： 定义判定函数 + 二分模板，区间进行二分搜索
- #1231. Divide Chocolate https://leetcode.com/problems/divide-chocolate [hard]
- #1011. Capacity to Ship packages within D days https://leetcode.com/problems/capacity-to-ship-packages-within-d-days [hard]
- #410. Split Array Largest Sum https://leetcode.com/problems/split-array-largest-sum [hard]
- #719. Find K-th Smallest Pair Distance https://leetcode.com/problems/find-k-th-smallest-pair-distance/ [hard] 结合双指针和二分

### 矩阵搜索 2D matrix 
- #378. Kth Samllest Element in a Sorted Matrix https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix 
- #74. Search a 2D Matrix https://leetcode.com/problems/search-a-2d-matrix
- #240. Search a 2D Matrix II https://leetcode.com/problems/search-a-2d-matrix-ii 








---
# Double pointers and sliding windows 双指针和滑动窗口

### 相向双指针
- #1. Two Sum https://leetcode.com/problems/two-sum
- #15. 3Sum https://leetcode.com/problems/3sum
- #75. Sort Colors https://leetcode.com/problems/sort-colors/ 
- #1229. Meeting Scheduler https://leetcode.com/problems/meeting-scheduler 
- #125. Valid Palindrome https://leetcode.com/problems/valid-palindrome

### 背向双指针
- #5. Longest Palindromic Substring https://leetcode.com/problems/longest-palindromic-substring
- #408. Valid Word Abbreviation https://leetcode.com/problems/valid-word-abbreviation
- #409. Longest Palindrome https://leetcode.com/problems/longest-palindrome 
- #680. Valid Palindrome II https://leetcode.com/problems/valid-palindrome-ii 

### Sliding windows 
- #3. Longest Substring Without Repeating Characters https://leetcode.com/problems/longest-substring-without-repeating-characters
- #76. Minimum Window Substring https://leetcode.com/problems/minimum-window-substring
- #1004. Max Consecutive Ones III https://leetcode.com/problems/max-consecutive-ones-iii
- #209. Minimum Size Subarray Sum https://leetcode.com/problems/minimum-size-subarray-sum
- #1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit 

> 新添加一些题
- #38. Count and Say https://leetcode.com/problems/count-and-say (用到拼接的思想，如何双指针)
- #30. Substring with Concatenation of All Words https://leetcode.com/problems/substring-with-concatenation-of-all-words (sliding window好题)
- #228. Summary Ranges https://leetcode.com/problems/summary-ranges/ 








---
# Sorting 排序
python自带的sorted 排序算法是timsort
> best time - o(n), average and worst case is o(nlogn);  space o(n); 
### Quick sort - time o(nlogn), space o(1)
> 先整体有序，再局部有序，利用分治的思想，递归的程序设计方式
### Merge sort - time o(nlogn), space o(n)
> 先局部有序，再整体有序

- #148. Sort List https://leetcode.com/problems/sort-list/ 
    > 超级麻烦，要有merge sort，感觉这种排序题在linked list 难度很大，而且容易出
- #179. Largest Number https://leetcode.com/problems/largest-number/
    > 不理解，官方答案超玄乎，但还是高频很多地方考过！
- #75. Sort Colors https://leetcode.com/problems/sort-colors/ 
- #493. Reverse Pairs https://leetcode.com/problems/reverse-pairs/ 
- #23. Merge k Sorted Lists  https://leetcode.com/problems/merge-k-sorted-lists/ 










---
# Array and string 相关题

### In-place 要求O(1)处理的题
- #26. Remove Duplicates from Sorted Array https://leetcode.com/problems/remove-duplicates-from-sorted-array
- #283. Move Zeroes https://leetcode.com/problems/move-zeroes 
- #48. Rotate Image https://leetcode.com/problems/rotate-image [2D matrix]
- #189. Rotate Array https://leetcode.com/problems/rotate-array
- #41. First Missing Positive https://leetcode.com/problems/first-missing-positive [hard]

### 安排会议 
- #252. Meeting Rooms https://leetcode.com/problems/meeting-rooms/ 
- #253. Meeting Rooms II https://leetcode.com/problems/meeting-rooms-ii/ [heap也可以解]
- #1094. Car Pooling https://leetcode.com/problems/car-pooling/ [prefix sum 差分]
- #452. Minimum Number of Arrows to Burst Balloons https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/ 
- #1235. Maximum Profit in Job Scheduling https://leetcode.com/problems/maximum-profit-in-job-scheduling/ [hard]
- #2054. Two Best Non-Overlapping Events https://leetcode.com/problems/two-best-non-overlapping-events/ 

### 区间操作：插入，合并，删除，非overlap
- #56. Merge Intervals https://leetcode.com/problems/merge-intervals/ [前缀和prefix sum]
- #57. Insert Interval https://leetcode.com/problems/insert-interval/ 
- #1272. Remove Interval https://leetcode.com/problems/remove-interval/ 
- #435. Non-overlapping Intervals https://leetcode.com/problems/non-overlapping-intervals/ 

### Subsequence 很多题都没做过！
- #392. Is Subsequence https://leetcode.com/problems/is-subsequence 
- #792. Number of Matching Subsequences https://leetcode.com/problems/number-of-matching-subsequences 
- #727. Minimum Window Subsequence https://leetcode.com/problems/minimum-window-subsequence [hard]
- #300. Longest Increasing Subsequence https://leetcode.com/problems/longest-increasing-subsequence 

### Subarray/substring 连续的，也有很多题
- #1062. Longest Repeating Substring https://leetcode.com/problems/longest-repeating-substring
- #1044. Longest Duplicate Substring https://leetcode.com/problems/longest-duplicate-substring [hard]
- #5. Longest Palindromic Substring https://leetcode.com/problems/longest-palindromic-substring 
- #3. Longest Substring Without Repeating Characters https://leetcode.com/problems/longest-substring-without-repeating-characters

### Subset 
- #78. Subsets https://leetcode.com/problems/subsets 
- #90. Subsets II https://leetcode.com/problems/subsets-ii 
- #368. Largest Divisible Subset https://leetcode.com/problems/largest-divisible-subset 

### prefix sum 前缀和
> 基本题，前缀和差分
- #303. Range Sum Query - Immutable https://leetcode.com/problems/range-sum-query-immutable 
- #304. Range Sum Query 2D - Immutable https://leetcode.com/problems/range-sum-query-2d-immutable 

> subarry类型
- #53. Maximum Subarray https://leetcode.com/problems/maximum-subarray 
- #325. Maximum Size Subarray Sum Equals k https://leetcode.com/problems/maximum-size-subarray-sum-equals-k 
- #525. Contiguous Array  https://leetcode.com/problems/contiguous-array
- #560. Subarray Sum Equals K https://leetcode.com/problems/subarray-sum-equals-k 
- #1248. Count Number of Nice Subarrays https://leetcode.com/problems/count-number-of-nice-subarrays 

> key是前缀和mod k的余数
- #523. Continuous Subarray Sum https://leetcode.com/problems/continuous-subarray-sum
- #974. Subarray Sums Divisible by K https://leetcode.com/problems/subarray-sums-divisible-by-k
- #1590. Make Sum Divisible by P https://leetcode.com/problems/make-sum-divisible-by-p
- #1524. Number of Sub-arrays With Odd Sum https://leetcode.com/problems/number-of-sub-arrays-with-odd-sum 

>前缀积， 前缀后缀信息同时使用
- #152. Maximum Product Subarray https://leetcode.com/problems/maximum-product-subarray
- #238. Product of Array Except Self https://leetcode.com/problems/product-of-array-except-self
- #724. Find Pivot Index https://leetcode.com/problems/find-pivot-index 

>差分方法，很好用
- #370. Range Addition https://leetcode.com/problems/range-addition 
- #1109. Corporate Flight Bookings https://leetcode.com/problems/corporate-flight-bookings 
- #1854. Maximum Population Year https://leetcode.com/problems/maximum-population-year















---
# Linked list 链表

### 快慢双指针操作（detect cycle, get middle, get kth element）
- #141. Linked List Cycle https://leetcode.com/problems/linked-list-cycle
- #19. Remove Nth Node From End of List https://leetcode.com/problems/remove-nth-node-from-end-of-list 

### 翻转链表 reverse linked list (dummy head)
- #206. Reverse Linked List https://leetcode.com/problems/reverse-linked-list 
- #25. Reverse Nodes in k-Group https://leetcode.com/problems/reverse-nodes-in-k-group [hard]

### LRU/LFU
- #146. LRU Cache https://leetcode.com/problems/lru-cache 
- #460. LFU Cache https://leetcode.com/problems/lfu-cache [hard]

### Deep copy
- #138. Copy List with Random Pointer https://leetcode.com/problems/copy-list-with-random-pointer 

### Merge LinkedList 
- #2. Add Two Numbers https://leetcode.com/problems/add-two-numbers 










---
# Tree 树

### tree 的遍历traverse (in-order, pre-order, post-order)
- #314. Binary Tree Vertical Order Traversal https://leetcode.com/problems/binary-tree-vertical-order-traversal 
- #199. Binary Tree Right Side View https://leetcode.com/problems/binary-tree-right-side-view

### 递归方法 
- #124. Binary Tree Maximum Path Sum https://leetcode.com/problems/binary-tree-maximum-path-sum [hard]
- #366. Find Leaves of Binary Tree https://leetcode.com/problems/find-leaves-of-binary-tree 

### Lowest Common Ancestor 系列
- #236. Lowest Common Ancestor of a Binary Tree https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree
- #235. Lowest Common Ancestor of a Binary Search Tree https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree 
- #1650. Lowest Common Ancestor of a Binary Tree III https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii
- #1644. Lowest Common Ancestor of a Binary Tree II https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-ii
- #1123. Lowest Common Ancestor of Deepest Leaves https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves
- #1676. Lowest Common Ancestor of a Binary Tree IV https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iv 

### Binary Search Tree (BST 性质，中序遍历increasing)
- #98. Validate Binary Search Tree https://leetcode.com/problems/validate-binary-search-tree 

### 编码解码 
- #449. Serialize and Deserialize BST  https://leetcode.com/problems/serialize-and-deserialize-bst/ 
- #297. Serialize and Deserialize Binary Tree  https://leetcode.com/problems/serialize-and-deserialize-binary-tree/ 

### 把tree变成graph
- #863. All Nodes Distance K in Binary Tree https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree 










---
# Stack，Queue and Heap 栈，队列 和堆

### Stack 
> 括号题 (括号补充题:20,22,32,301,678)
- #921. Minimum Add to Make Parentheses Valid https://leetcode.com/problems/minimum-add-to-make-parentheses-valid 
- #1249. Minimum Remove to Make Valid Parentheses https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses 

> 计算器系列题
- #227. Basic Calculator II https://leetcode.com/problems/basic-calculator-ii 
- #224. Basic Calculator https://leetcode.com/problems/basic-calculator [hard]
- #770. Basic Calculator IV https://leetcode.com/problems/basic-calculator-iv [hard]
- #772. Basic Calculator III https://leetcode.com/problems/basic-calculator-iii [hard]

> nested list iterator 系列
- #339. Nested List Weight Sum https://leetcode.com/problems/nested-list-weight-sum 
- #341. Flatten Nested List Iterator https://leetcode.com/problems/flatten-nested-list-iterator 
- #364. Nested List Weight Sum II https://leetcode.com/problems/nested-list-weight-sum-ii 

> others
- #394. Decode String https://leetcode.com/problems/decode-string 
- #726. Number of Atoms https://leetcode.com/problems/number-of-atoms [hard]

> 单调栈一般都是optimal solution，但首先用brute force！
- #496. Next Greater Element I  https://leetcode.com/problems/next-greater-element-i/ 
- #503. Next Greater Element II https://leetcode.com/problems/next-greater-element-ii/
- #556. Next Greater Element III https://leetcode.com/problems/next-greater-element-iii/ (不太好想！)
- #739. Daily Temperatures https://leetcode.com/problems/daily-temperatures/ 
- #901. Online Stock Span https://leetcode.com/problems/online-stock-span/
- #316. Remove Duplicate Letters https://leetcode.com/problems/remove-duplicate-letters/ 
- #402. Remove K Digits https://leetcode.com/problems/remove-k-digits/ 
- #581. Shortest Unsorted Continuous Subarray https://leetcode.com/problems/shortest-unsorted-continuous-subarray/ (可以用但不是必须用，sorting的解可以接受）
- #2104. Sum of Subarray Ranges https://leetcode.com/problems/sum-of-subarray-ranges/ (单调栈的解非常复杂，虽然能优化！)
- #1762. Buildings With an Ocean View https://leetcode.com/problems/buildings-with-an-ocean-view/ 

> 接雨水还有histogram专题
- #84. Largest Rectangle in Histogram https://leetcode.com/problems/largest-rectangle-in-histogram/ 
- #907. Sum of Subarray Minimums https://leetcode.com/problems/sum-of-subarray-minimums/ 
- #42. Trapping Rain Water https://leetcode.com/problems/trapping-rain-water/ 
- #11. Container With Most Water https://leetcode.com/problems/container-with-most-water/ 

> 4个类似的单调栈  https://leetcode-cn.com/problems/remove-duplicate-letters/solution/yi-zhao-chi-bian-li-kou-si-dao-ti-ma-ma-zai-ye-b-4/ 


### Queue (BFS相关提，单独的比较少)
- #239. Sliding Window Maximum https://leetcode.com/problems/sliding-window-maximum 
- #346. Moving Average from Data Stream https://leetcode.com/problems/moving-average-from-data-stream 

### Heap 很多难题可以用heap， 很多hard
> top k 问题
- #215. Kth Largest Element in an Array https://leetcode.com/problems/kth-largest-element-in-an-array
- #347. Top K Frequent Elements https://leetcode.com/problems/top-k-frequent-elements

> 中位数问题
- #295. Find Median from Data Stream https://leetcode.com/problems/find-median-from-data-stream
- #4. Median of Two Sorted Arrays https://leetcode.com/problems/median-of-two-sorted-arrays [hard]










---
# Graph 图

### Graph traverse - BFS,DFS 模板
> 基本模板，有没有基本题呢？

### 矩阵图， 4周neighbor相连 
> 0-1 islands 系列
- #200. Number of Islands https://leetcode.com/problems/number-of-islands
- #305. Number of Islands II https://leetcode.com/problems/number-of-islands-ii [hard]
- #694. Number of Distinct Islands https://leetcode.com/problems/number-of-distinct-islands 
- #711. Number of Distinct Islands II https://leetcode.com/problems/number-of-distinct-islands-ii [hard]
- #1254. Number of Closed Islands https://leetcode.com/problems/number-of-closed-islands

> world search 系列
- #79. Word Search https://leetcode.com/problems/word-search
- #212. Word Search II https://leetcode.com/problems/word-search-ii [hard]

> others
- #417. Pacific Atlantic Water Flow https://leetcode.com/problems/pacific-atlantic-water-flow 
- 690. Employee Importance https://leetcode.com/problems/employee-importance 

### Data(state)看成node,operation变成edge - 很多时候变成动态规划
- #127. Word Ladder https://leetcode.com/problems/word-ladder [hard]
- #126. Word Ladder II https://leetcode.com/problems/word-ladder-ii [hard]
- #1345. Jump Game IV https://leetcode.com/problems/jump-game-iv 

### 拓扑排序topological sort 
- #269. Alien Dictionary https://leetcode.com/problems/alien-dictionary [hard]
- #310. Minimum Height Trees https://leetcode.com/problems/minimum-height-trees
- #366. Find Leaves of Binary Tree https://leetcode.com/problems/find-leaves-of-binary-tree
- #444. Sequence Reconstruction https://leetcode.com/problems/sequence-reconstruction

> 排课系列
- #207. Course Schedule https://leetcode.com/problems/course-schedule
- #210. Course Schedule II https://leetcode.com/problems/course-schedule-ii 
- #630. Course Schedule III https://leetcode.com/problems/course-schedule-iii [hard]
- #1462. Course Schedule IV https://leetcode.com/problems/course-schedule-iv [hard]

### 图是否有cycle和二分染色
- #785. Is Graph Bipartite? https://leetcode.com/problems/is-graph-bipartite
- #1192. Critical Connections in a Network https://leetcode.com/problems/critical-connections-in-a-network 

### 最长最短路径 - BFS
- #994. Rotting Oranges https://leetcode.com/problems/rotting-oranges 
- #909. Snakes and Ladders https://leetcode.com/problems/snakes-and-ladders 
- #1091. Shortest Path in Binary Matrix https://leetcode.com/problems/shortest-path-in-binary-matrix 
- #1293. Shortest Path in a Grid with Obstacles Elimination https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination 


### Backtracking + DFS + memo 
- #526. Beautiful Arrangement https://leetcode.com/problems/beautiful-arrangement 
- #22. Generate Parentheses https://leetcode.com/problems/generate-parentheses 

### BFS + binary search 
- #1102. Path With Maximum Minimum Value https://leetcode.com/problems/path-with-maximum-minimum-value 
- #778. Swim in Rising Water https://leetcode.com/problems/swim-in-rising-water [hard]


### Word 系列 
- #139. Word Break https://leetcode.com/problems/word-break 
- #140. Word Break II https://leetcode.com/problems/word-break-ii 
- #290. Word Pattern https://leetcode.com/problems/word-pattern 
- #291. Word Pattern II https://leetcode.com/problems/word-pattern-ii 







---
# DFS + backtracking 

### 排列组合系列
- #46. Permutations https://leetcode.com/problems/permutations 
- #47. Permutations II https://leetcode.com/problems/permutations-ii/
- #31. Next Permutation https://leetcode.com/problems/next-permutation
- #77. Combinations https://leetcode.com/problems/combinations 
- #78. Subsets https://leetcode.com/problems/subsets/
- #90. Subsets IIhttps://leetcode.com/problems/subsets-ii/
- #39. Combination Sum https://leetcode.com/problems/combination-sum/
- #40. Combination Sum II https://leetcode.com/problems/combination-sum-ii/
- #60. Permutation Sequence https://leetcode.com/problems/permutation-sequence/
- #131. Palindrome Partitioning https://leetcode.com/problems/palindrome-partitioning/
- #267. Palindrome Permutation II https://leetcode.com/problems/palindrome-permutation-ii/ 








---
# DP 动态规划

### Jump Game 系列 


 -->


