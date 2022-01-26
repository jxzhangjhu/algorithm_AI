### 使用条件
1. 滑动窗口 - 90% 的概率
2. 时间复杂度要求 O(n) - 80%的概率
3. 要求原地操作，只可以交换使用，不能使用额外空间，所以空间复杂度O(1) - 80% 
4. 有子数组subarray， 子字符串substring的关键词 - 50%
5. 有回文问题 palindrome 关键词 - 50% 

### time complexity
>> 时间复杂度与最内层循环主体的loop执行次数有关， 与有多少重循环无关
### space complexity
>> 只需要分配2个指针的额外内存，所以space 是O(1)


## 相向双指针 - patition in quicksort 

```python
class Solution:
    def patition(self, A, start, end):
        if start >= end:
            return 

        left, right = start, end
        # key point 1: pivot is the value, not the index 
        pivot = A[(start + end) // 2]
        # key point 2: every time you compare left & right, it should be left <= right not left < right 
        while left <= right:
            while left <= right and A[left] < pivot:
                left += 1
            while left <= right and A[right] > pivot: 
                right -= 1
            if left <= right:
                A[left], A[right] = A[right], A[left]
                left += 1
                right -= 1
```

## 背向双指针 

```python
class Solution:
    def patition(self, A, start, end):

        left = position 
        right = position + 1
        while left >=0 and right < len(s):
            if left and right 可以停下来了：
                break

            left -= 1
            right += 1 


```

## 同向双指针 - 快慢指针 

```python
class Solution:
    def patition(self, A, start, end):

        j = 0
        for i in range(n):
            # 不满足则循环到满足搭配为止

            while j < n and i and j 之间不满足条件:
                j += 1

            if i到j之间满足条件:
                处理i到j这段区间 

```

#### Example: 209. Minimum Size Subarray Sum
https://leetcode.com/problems/minimum-size-subarray-sum/ 

Given an array of positive integers nums and a positive integer target, return the minimal length of a contiguous subarray [numsl, numsl+1, ..., numsr-1, numsr] of which the sum is greater than or equal to target. If there is no such subarray, return 0 instead.
Example 1:
```
Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: The subarray [4,3] has the minimal length under the problem constraint.
```

Example 2:
```
Input: target = 4, nums = [1,4,4]
Output: 1
```

Example 3:
```
Input: target = 11, nums = [1,1,1,1,1,1,1,1]
Output: 0
```

Constraints:

1 <= target <= 109
1 <= nums.length <= 105
1 <= nums[i] <= 105
 

Follow up: If you have figured out the O(n) solution, try coding another solution of which the time complexity is O(n log(n)).

>Solution

```python 
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        
        # 同向双指针的问题 time 可以做到o(n)
        if sum(nums) < target:
            return 0
        
        fast = 0 
        length = len(nums)
        res = length # 注意这个用这个长度就可以
        
        for slow in range(length):
            while fast < length and sum(nums[slow:fast]) < target: 
                fast += 1
            if sum(nums[slow:fast]) >= target:
                res = min(fast - slow, res)  # 不断迭代
                
        return res
 ```

## 合并双指针 

```python
class Solution:
    def merge(self, A, start, end):
        new_list = [] 
        i, j = 0, 0

        # 合作的过程只能操作i, j 的移动， 不要去list.pop(0) 之类的操作, 因为pop(0) 是O(n)的时间复杂度在python
        while i < len(list1) and j < len(list2):
            if list1[i] < list[j]:
                new_list.append(list1[i]) 
                i += 1
            else:
                new_list.append(list2[j])
                j += 1

            # 合并剩下的数到new_list里
            # 不要用new_list.extend(list1[i:])之类的方法
            # 因为list1[i:] 会产生额外的空间消耗

            while i < len(list1):
                new_list.append(list1[i])
                i += 1
            while j < len(list2):
                new_list.append(list2[j])
                j += 1

            return new_list 



```
