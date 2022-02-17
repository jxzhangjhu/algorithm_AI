# linkedlist 

## 常规操作
1. head = head.next 这是指针后移，把当前node往后挪 
2. head.next = anothernode or curr.next 都是改变node的指针，重新指向新的node，一旦发生，不可改变

### Example - 219 · Insert Node in Sorted Linked List 
Example
Example 1:
```
Input: head = 1->4->6->8->null, val = 5
Output: 1->4->5->6->8->null
```
Example 2:
```
Input: head = 1->null, val = 2
Output: 1->2->null
```

>Solution
```python
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: The head of linked list.
    @param val: An integer.
    @return: The head of new linked list.
    """
    def insertNode(self, head, val):
        # write your code here

        dummy = ListNode(next=head,val=None)
        curr_node = dummy
        while curr_node.next and curr_node.next.val <= val:
            curr_node = curr_node.next # node 指针后移

        # add new node 
        new_node = ListNode(val)
        # 左右链接
        # left 
        # curr_node.next = new_node #先把前面连上
        # new_node.next = curr_node.next # 再把后面连上，这样就成环了！

        new_node.next = curr_node.next # 先把后面连上 ,next -> node
        curr_node.next = new_node # 再把前面连上 

        return dummy.next
```

### Example - 452 · Remove Linked List Elements 
Example
Example 1:
```
Input: head = 1->2->3->3->4->5->3->null, val = 3
Output: 1->2->4->5->null
```
Example 2:
```
Input: head = 1->1->null, val = 1
Output: null
```

>Solution
```python
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: a ListNode
    @param val: An integer
    @return: a ListNode
    """
    def removeElements(self, head, val):
        # write your code here

        dummy = ListNode(next=head,val=None)
        curr_node = dummy
        while curr_node.next:
            if curr_node.next.val == val:
                curr_node.next = curr_node.next.next
            else:
                curr_node = curr_node.next

        return dummy.next 
```

### Example - 228 · Middle of Linked List

Description
Find the middle node of a linked list and return it.

Example
Example 1:
```
Input:  1->2->3
Output: 2   
Explanation: return the middle node.
```
Example 2:
```
Input:  1->2
Output: 1   
Explanation: If the length of list is even return the center left one.
```

>Solution
```python
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: the head of linked list.
    @return: a middle node of the linked list
    """
    def middleNode(self, head):
        # write your code here
        
        if not head:
            return 
        slow = head
        fast = head

        while fast.next and fast.next.next:
             slow = slow.next
             fast = fast.next.next 

        return slow
```


### Example - 61. Rotate List 
Given the head of a linked list, rotate the list to the right by k places. 
```
Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]
```
>Solution 
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head:
            return 
        dummy = ListNode(next=head,val=None)
        
        # 计算长度
        def get_length(head):
            length = 0
            while head:
                length += 1
                head = head.next 
            return length 
        
        # 避免重复计算
        length = get_length(head)
        k = k % length
        
        # 先copy过来从dummy
        ahead = dummy
        behind = dummy
        
        # 先move k 步，把ahead 挪过来，现在ahead 是3，4，5
        for i in range(k):
            ahead = ahead.next
    
        # 调整ahead 和 behind的头位置
        while ahead.next: # 往前move，就是把前面的都cut掉了？ 都是从dummy copy过来的！
            behind = behind.next
            ahead = ahead.next
        
        # dummy 永远是0， dummy.next 就是1 
        # 右侧的永远是node，左侧如果是node，那么就是update node， 如果是next，那就是update新的指向， ahead 指向1
        ahead.next = dummy.next # 5 -> 1
        dummy.next = behind.next # dummy -> 4 - > 5 
        behind.next = None # current behind is 3, cut 3 next # 没有这个肯定不行
        print(dummy.next)
        # 这些node 都在空间存在，实际上就是在找他们的之间的链接，可能有多个，但是就会报错！必须1对1，检查的时候也是看这些node，都在对应谁！
        return dummy.next
        
```

### Example - 19. Remove Nth Node From End of List 
Given the head of a linked list, remove the nth node from the end of the list and return its head. could you finish it by one pass? 

>Solution
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        
#         # 暴力解  能够accept 但不是one pass - two pass 
#         dummy = ListNode(next = head, val = None)
#         curr = dummy 
#         length = 0 
#         while curr.next:
#             curr = curr.next 
#             length += 1
            
#         new = dummy
#         for i in range(length - n):
#             new = new.next 
#         new.next = new.next.next 
#         return dummy.next 
    
#     # time - o(L), space o(1)
        
        # one pass 类似middel那个题，就是两个node，然后第一个先跑，让两者间隔n，那么第二个到头了，第一个的位置就是倒数第n个
        
        dummy = ListNode(next = head, val = None)
        
        slow = dummy
        fast = dummy 
        
        for i in range(n):
            fast = fast.next 
            
        while fast.next:
            slow = slow.next
            fast = fast.next 
            
        slow.next = slow.next.next 
        
        return dummy.next 

    # time - o(L), space - o(1)
```

### Example - 142. Linked List Cycle II 

Given the head of a linked list, return the node where the cycle begins. If there is no cycle, return null.

>Solution
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        # 这个题还可以
        dummy = ListNode(next = head, val = None)
        slow = dummy.next
        fast = dummy.next
        
        while fast and fast.next:
            slow = slow.next 
            fast = fast.next.next
            
            # slow 和 fast 相遇，还得看slow 和 head的情况
            if slow == fast:
                p = head
                q = slow
                while p != q:
                    p = p.next
                    q = q.next
                    
                return p
            
        return None
```


---

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