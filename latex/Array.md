# General array

## 很多题结合sorting，还有hashmap来做

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