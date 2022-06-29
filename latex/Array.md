# General array

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
非常easy， 主要是和hashtable 一起用 

### 169. Majority Element  https://leetcode.com/problems/majority-element/ 
### 229. Majority Element II 
### 1150. Check If a Number Is Majority Element in a Sorted Array 

