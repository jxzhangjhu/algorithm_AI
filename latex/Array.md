# General array

✅✅✅ 很多题结合sorting，双指针，hashmap ✅✅✅ 


### 977. Squares of a Sorted Array https://leetcode.com/problems/squares-of-a-sorted-array/

Given an integer array nums sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.
```
Example 1:

Input: nums = [-4,-1,0,3,10]
Output: [0,1,9,16,100]
Explanation: After squaring, the array becomes [16,1,0,9,100].
After sorting, it becomes [0,1,9,16,100].
Example 2:

Input: nums = [-7,-3,2,3,11]
Output: [4,9,9,49,121]
```

***Follow up: Squaring each element and sorting the new array is very trivial, could you find an O(n) solution using a different approach?***

```python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        # way 1 暴力干，但是时间复杂度是o(nlogn) 取决于不同的内部算法implement details 这个需要了解一下！
        # way 2 - double pointer 
        # 这个题挺有意思，需要利用nums已经排序的信息，从大往小排，因为两端的应该最大，然后中间的可能比较小，如果传统的双指针，很难从小开始排
        res = [-1]*len(nums)
        left, right = 0, len(nums) - 1
        k = len(nums) - 1
        while left <= right: 
            if abs(nums[left]) <= abs(nums[right]) and left <= right:
                square = nums[right] ** 2
                right -= 1
            
            elif left <= right:
                square = nums[left] ** 2
                left += 1
        
            res[k] = square
            k -= 1
            
        return res     
```

### 88. Merge Sorted Array https://leetcode.com/problems/merge-sorted-array/ 

You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.

Merge nums1 and nums2 into a single array sorted in non-decreasing order.

The final sorted array should not be returned by the function, but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.
```
Example 1:

Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.
Example 2:

Input: nums1 = [1], m = 1, nums2 = [], n = 0
Output: [1]
Explanation: The arrays we are merging are [1] and [].
The result of the merge is [1].
Example 3:

Input: nums1 = [0], m = 0, nums2 = [1], n = 1
Output: [1]
Explanation: The arrays we are merging are [] and [1].
The result of the merge is [1].
Note that because m = 0, there are no elements in nums1. The 0 is only there to ensure the merge result can fit in nums1.
```
***Follow up: Can you come up with an algorithm that runs in O(m + n) time?***

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
        Output: [1,2,2,3,5,6]
        """
        # 暴力干，但不是要考的内容
        # nums1[m:n+m] = nums2
        # nums1.sort()
        # time o(n+m),space o(1) 这是错的
        # 实际上的时间复杂度o((n+m)log(n+m)), 所以这个题要求o(n+m), space o(log(n+m)) 快排的平均空间复杂度
        
        # double pointer  - small to big 
        # 这个可以单独开一个空间，o(n+m)然后再赋值给nums1，开始以为只能在nums1上操作，就比较麻烦
        # 充分利用nums1 and nums2 都已经排序的优点
        
#         p1, p2 = 0, 0
#         res = []
#         while p1 < m or p2 < n: # 两者必须都不满足才停下来
#             if p1 == m:
#                 res.append(nums2[p2])
#                 p2 += 1
#             elif p2 == n:
#                 res.append(nums1[p1])
#                 p1 += 1
#             elif nums1[p1] < nums2[p2]: # 不能先判定这个，就会越界，这个多重比较的时候，如何选择判断条件，之前遇到过类似的问题！
#                 res.append(nums1[p1])
#                 p1 += 1
#             else:
#                 res.append(nums2[p2])
#                 p2 += 1
                
#         nums1[:] = res
        
        # 这个时间复杂度是o(m+n)， 但是空间是o(m+n)， 因为需要重新开一个数组，然后再赋值，耗费空间了！
        # 类似这个题Squares of a Sorted Array 
        # double pointer from big to small 不需要开额外空间，之前也有这样一个技巧！
        p1, p2 = m - 1, n - 1
        tail = m + n - 1
        while p1 >= 0 or p2 >= 0:
            if p1 == -1:
                nums1[tail] = nums2[p2]
                p2 -= 1
            elif p2 == -1:
                nums1[tail] = nums1[p1]
                p1 -= 1
            elif nums1[p1] > nums2[p2]:
                nums1[tail] = nums1[p1]
                p1 -= 1
            else:
                nums1[tail] = nums2[p2]
                p2 -= 1
            tail -= 1
        
        # time o(m+n), 移动指针单调递减，最多移动m+n次
        # space，因为是对数组原地修改，不需要额外空间o(1)
```



### 360. Sort Transformed Array https://leetcode.com/problems/sort-transformed-array/ 
Given a sorted integer array nums and three integers a, b and c, apply a quadratic function of the form f(x) = ax2 + bx + c to each element nums[i] in the array, and return the array in a sorted order.
```
Example 1:

Input: nums = [-4,-2,2,4], a = 1, b = 3, c = 5
Output: [3,9,15,33]
Example 2:

Input: nums = [-4,-2,2,4], a = -1, b = 3, c = 5
Output: [-23,-5,1,7]

```
***Follow up: Could you solve it in O(n) time?***

```python
# class Solution:
#     def sortTransformedArray(self, nums: List[int], a: int, b: int, c: int) -> List[int]:
#         # brute force, time o(nlogn), space o(n)
#         res = []
#         for num in nums:
#             f = a*num**2 + b*num + c
#             res.append(f)
#         return sorted(res)
class Solution:
    def sortTransformedArray(self, nums: List[int], a: int, b: int, c: int) -> List[int]:
        def quadratic(x):
            return a*x*x + b*x + c 
        n = len(nums)
        index = 0 if a < 0 else n-1
        l, r, ans = 0, n-1, [0] * n
        while l <= r:
            l_val, r_val = quadratic(nums[l]), quadratic(nums[r])
            if a >= 0:
                if l_val > r_val:
                    ans[index] = l_val 
                    l += 1
                else:    
                    ans[index] = r_val 
                    r -= 1
                index -= 1
            else:
                if l_val > r_val:
                    ans[index] = r_val 
                    r -= 1
                else:    
                    ans[index] = l_val 
                    l += 1
                index += 1
        return ans
``` 







### 628. Maximum Product of Three Numbers https://leetcode.com/problems/maximum-product-of-three-numbers/ 
Given an integer array nums, find three numbers whose product is maximum and return the maximum product.
```
Example 1:

Input: nums = [1,2,3]
Output: 6
Example 2:

Input: nums = [1,2,3,4]
Output: 24
Example 3:

Input: nums = [-1,-2,-3]
Output: -6
```
```python
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        
        # 这个需要归纳出来啊，都是正数没问题 最后三个sort
        # case 2  1个负数，2个正数
        # case 3  2个负数， 1个正数
        # case 4, 3个负数
        
        # 确实比较boring
        
        nums.sort()
        n = len(nums)
        positive = nums[n-1] * nums[n-2] * nums[n-3]
        negative = nums[0] * nums[1] * nums[n-1] 
        return max(positive, negative)
    
        # time o(nlogn), space o(logn) 都是sorting带来的
```

✅✅✅ majority element 三个题 ✅✅✅  
非常easy， 主要是和hashtable 一起用, 很简单

### 169. Majority Element  https://leetcode.com/problems/majority-element/ 
### 229. Majority Element II 
### 1150. Check If a Number Is Majority Element in a Sorted Array 


✅✅✅ contains duplicate 三个题 ✅✅✅  
在sliding window 里面讨论过了

### 217. Contains Duplicate
### 219. Contains Duplicate II
### 220. Contains Duplicate III


✅✅✅ Meeting room 相关的几个题 ✅✅✅  

### 252. Meeting Rooms
Given an array of meeting time intervals where intervals[i] = [starti, endi], determine if a person could attend all meetings.
```
Example 1:

Input: intervals = [[0,30],[5,10],[15,20]]
Output: false
Example 2:

Input: intervals = [[7,10],[2,4]]
Output: true
```
> 简单sorting然后比较即可，后面几个类似的interval的题
```python
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        # 简单判断一下就行， time o(nlogn), space o(1) 
        intervals.sort()
        print(intervals) # 以第一个为key排列
        for i in range(len(intervals)-1):
            if  intervals[i+1][0] < intervals[i][1]:
                return False
        return True 
```


### 253. Meeting Rooms II 
Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required.
```
Example 1:

Input: intervals = [[0,30],[5,10],[15,20]]
Output: 2
Example 2:

Input: intervals = [[7,10],[2,4]]
Output: 1
```
> 可以用double pointer，也可以用heap

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        
        # heap 解法 - time o(nlogn), space o(n)
        intervals.sort(key=lambda x: x[0])
        print(intervals)
        h = []
        maxRooms = 0
        for i in range(len(intervals)):
            print(i)
            if not h:
                # initialize the heap with the first interval
                heappush(h, intervals[i][1])
                print(h)
            else:
                # keep popping from heap if the current start time is more than
                # top of the heap
                while h and intervals[i][0] >= h[0]:
                    heappop(h)
                # push the end time of this interval into the min-heap
                heappush(h, intervals[i][1])
                print(h)
                
            maxRooms = max(maxRooms, len(h))
        return maxRooms

class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
            
        #把区间变成2个数组：start时间数组和end时间数组，并对两个数组排序。
        start_time = sorted([i[0] for i in intervals])
        end_time = sorted([i[1] for i in intervals])
        
        #然后一个指针遍历start数组，另一个指针指向end数组。
        start_pointer = 0
        end_pointer = 0
        
        ##如果start时间小于end时间，房间数就加1，start时间加1，比较并记录出现过的最多房间数
        ##start时间大于等于end，则所需房间数就减1，end指针加1。
        minrooms = 0
        cntrooms = 0
        
        while start_pointer < len(intervals)
            if start_time[start_pointer] < end_time[end_pointer]:
                cntrooms += 1
                minrooms = max(minrooms, cntrooms)
                start_pointer += 1
            else:
                cntrooms -= 1
                end_pointer += 1
        return minrooms
``` 

### 1229. Meeting Scheduler https://leetcode.com/problems/meeting-scheduler/ 

Given the availability time slots arrays slots1 and slots2 of two people and a meeting duration duration, return the earliest time slot that works for both of them and is of duration duration.

If there is no common time slot that satisfies the requirements, return an empty array.

The format of a time slot is an array of two elements [start, end] representing an inclusive time range from start to end.

It is guaranteed that no two availability slots of the same person intersect with each other. That is, for any two time slots [start1, end1] and [start2, end2] of the same person, either start1 > end2 or start2 > end1.

```
Example 1:

Input: slots1 = [[10,50],[60,120],[140,210]], slots2 = [[0,15],[60,70]], duration = 8
Output: [60,68]
Example 2:

Input: slots1 = [[10,50],[60,120],[140,210]], slots2 = [[0,15],[60,70]], duration = 12
Output: []
```
> double pointer, 空间更优； 也可以用heap，来优化时间
```python
class Solution:
    def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
        slots1.sort()
        slots2.sort()
        i = 0
        j = 0
        while i < len(slots1) and j < len(slots2):
                #有交集（ # Let's check if A[i] intersects B[j]）
                l = max(slots1[i][0], slots2[j][0])
                r = min(slots1[i][1], slots2[j][1])
                if r - l >= duration:#交集满足开会需要
                    return [l, l + duration]
                # always move the one that ends earlier
                if slots1[i][1] < slots2[j][1]:#交集不满足开会需要，2还有剩余
                    i += 1
                else:
                    j += 1
        return []
``` 


### 56. Merge Intervals  https://leetcode.com/problems/merge-intervals/

Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

```
Example 1:

Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
Example 2:

Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.
```
> 两种解法，多比较，这个题非常经典, 都排序了
```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0]) # sorting o(nlogn)
        merged = []
        for interval in intervals:
            # 如果列表为空，或者当前区间与上一区间不重合，直接添加
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # 否则的话，我们就可以与上一区间进行合并
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key = lambda i: i[0]) # sorting o(nlogn)
        output = [intervals[0]]
        
        for start, end in intervals[1:]:
            lastEnd = output[-1][1]
            
            if start <= lastEnd:
                output[-1][1] = max(lastEnd, end)
            else:
                output.append([start,end])
            
        return output
``` 


### 57. Insert Interval  https://leetcode.com/problems/insert-interval/

You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return intervals after the insertion.
```
Example 1:

Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
Example 2:

Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
```
> 也是一个模拟，没用sorting
```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        # 官方这个答案比较厉害，time o(n), space o(1), 这个不容易些感觉！
        left, right = newInterval
        placed = False
        ans = list()
        for li, ri in intervals:
            if li > right:
                # 在插入区间的右侧且无交集
                if not placed:
                    ans.append([left, right])
                    placed = True
                ans.append([li, ri])
            elif ri < left:
                # 在插入区间的左侧且无交集
                ans.append([li, ri])
            else:
                # 与插入区间有交集，计算它们的并集
                left = min(left, li)
                right = max(right, ri)
        
        if not placed:
            ans.append([left, right])
        return ans

# 作者：LeetCode-Solution
# 链接：https://leetcode-cn.com/problems/insert-interval/solution/cha-ru-qu-jian-by-leetcode-solution/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
````


### 986. Interval List Intersections https://leetcode.com/problems/interval-list-intersections/ 

You are given two lists of closed intervals, firstList and secondList, where firstList[i] = [starti, endi] and secondList[j] = [startj, endj]. Each list of intervals is pairwise disjoint and in sorted order.

Return the intersection of these two interval lists.

A closed interval [a, b] (with a <= b) denotes the set of real numbers x with a <= x <= b.

The intersection of two closed intervals is a set of real numbers that are either empty or represented as a closed interval. For example, the intersection of [1, 3] and [2, 4] is [2, 3].
 
```
Example 1:
Input: firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
Example 2:

Input: firstList = [[1,3],[5,9]], secondList = []
Output: []
```

```python
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        res = []
        i = 0
        j = 0
        
        while i < len(firstList) and j < len(secondList):
               #有交集（ # Let's check if A[i] intersects B[j]）
                l = max(firstList[i][0], secondList[j][0])
                r = min(firstList[i][1], secondList[j][1])
                # if r - l >= duration:#交集满足开会需要
                if l <= r:
                    res.append([l, r])
                #always move the one that ends earlier
                if firstList[i][1] < secondList[j][1]:#交集不满足开会需要，2还有剩余
                    i += 1
                else:
                    j += 1
        return res
```

### 1094. Car Pooling https://leetcode.com/problems/car-pooling/ 
There is a car with capacity empty seats. The vehicle only drives east (i.e., it cannot turn around and drive west).

You are given the integer capacity and an array trips where trips[i] = [numPassengersi, fromi, toi] indicates that the ith trip has numPassengersi passengers and the locations to pick them up and drop them off are fromi and toi respectively. The locations are given as the number of kilometers due east from the car's initial location.

Return true if it is possible to pick up and drop off all passengers for all the given trips, or false otherwise.
```
Example 1:

Input: trips = [[2,1,5],[3,3,7]], capacity = 4
Output: false
Example 2:

Input: trips = [[2,1,5],[3,3,7]], capacity = 5
Output: true 
``` 
> 和meeting room 2 很像，一类题

```python
class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
    
        f = [0 for _ in range(1002)]
        for x, L, R in trips:       #差分统计的思想   台阶，上下楼梯的思想
            f[L] += x
            f[R] -= x
            
        if f[0] > capacity:
            return False
        for i in range(1, 1002):
            f[i] = f[i-1] + f[i]
            if f[i] > capacity:
                return False
        
        return True

# 思路简单：
# 差分统计每个站点的频次
# 1.一个数组f记录每个点的变化，初始化为0
# 2.遍历trips数组
# 对于一个 （人数，上车点，下车点）
# f[上车点] += 人数
# f[下车点] -= 人数
# 3.整理
# f[i] 是在f[i-1]的基础上，
# 在f[i-1]是在i-1这个点，车上有几个人 f[i]是记录i点的增减的人数
# f[i-1] + f[i] 是此时i点的人数 赋给f[i]
# 4.时时比较每个点（i）车上的人数，是否超过了capacity

# 作者：QRhqcDD90G
# 链接：https://leetcode-cn.com/problems/car-pooling/solution/cpython3java-chao-ji-qing-xi-jian-ji-cha-hnl1/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```


### 要做的几个题 06292022 涉及interval 的

### 452. Minimum Number of Arrows to Burst Balloons https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/
### 435. Non-overlapping Intervals https://leetcode.com/problems/non-overlapping-intervals/ 
### 495. Teemo Attacking https://leetcode.com/problems/teemo-attacking/ 

### 616. Add Bold Tag in String https://leetcode.com/problems/add-bold-tag-in-string/ 
### 763. Partition Labels https://leetcode.com/problems/partition-labels/ 


