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