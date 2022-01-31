# heap 

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


## time complexity
## space complexity


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

## 几种类型的双指针及相关题目

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


