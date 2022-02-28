## 使用条件
1. 排序数组（30%-40% 是二分法）
2. 当面试官要求找一个比o(n)更小的时间复杂度算法的时候，99% 就是二分logn
3. 找到数组中的一个分割位置，使得左半部分满足某个条件，右半部分不满足 100% 就是二分
4. 找到一个最大、最小的值使得某个条件被满足 90% 

## time complexity
>> o(logn)
## space complexity
>> o(1)

## 几种类型的双指针及相关题目


--- 
## template 

```python
class Solution:
    def binary_search(self, nums, target):

        # corner case 处理 - 这里等价于num is None or len(num) == 0
        if not nums: return -1 

        # 初始化
        start, end = 0, len(nums) - 1

        # 用start + 1 < end 而不是 start < end 的目的是为了避免死循环
        # the first position of target 的情况下不会出现死循环，但是在last position of target 的情况下会出现死循环
        # example, nums = [1, 1]， target = 1 

        while start + 1 < end:
            mid = start + (end - start) // 2 
            # or mid = (start + end) // 2

            # > = < 的逻辑先分开写，然后在看看 = 的情况是否能合并到其他分支里
            if nums[mid] < target: 
                start = mid 

            elif nums[mid] == target:
                end = mid 

            else: 
                end = mid 

            # 因为上面的循环退出条件是 start + 1 < end 
            # 因此这里循环结束的时候， start 和 end 的关系是相邻关系 (1和2，3和4 这种)
            # 因此需要再单独判断 start 和 end 这两个数谁是我们想要的答案
            # 如果是找 first position of target 就先看start， 否则就先看end 

            if nums[start] == target:
                return start 

            if nums[end] == target: 
                return end 

        return -1 

```


### Example - 162. Find Peak Element  (不用改code，可解852)
https://leetcode.com/problems/find-peak-element/ 

A peak element is an element that is strictly greater than its neighbors.

Given an integer array nums, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.

You may imagine that nums[-1] = nums[n] = -∞.

You must write an algorithm that runs in O(log n) time.


Example 1:
```
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.
```
Example 2:
```
Input: nums = [1,2,1,3,5,6,4]
Output: 5
Explanation: Your function can return either index number 1 where the peak element is 2, or index number 5 where the peak element is 6.
```

>Solution 
```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        
        if not nums:
            return -1
        
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            
            if nums[mid] > nums[mid + 1]:
                end = mid 
            else:
                start = mid 
               
        # 这步不在while 循环内
        if nums[start] > nums[end]:
            return start 
        else:
            return end
        
        # 这个题还可以改一下就是九章那个，not return index, instead the max number
        # return max(nums[start], nums[end])
            
        return -1 

    # time o(logn) and space o(1) 
```



