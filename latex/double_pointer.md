# Double pointer and sliding window 

## 使用条件
1. 滑动窗口 - 90% 的概率
2. 时间复杂度要求 O(n) - 80%的概率
3. 要求原地操作，只可以交换使用，不能使用额外空间，所以空间复杂度O(1) - 80% 
4. 有子数组subarray, 子字符串substring的关键词 - 50%
5. 有回文问题 palindrome 关键词 - 50% 

## time complexity
>> 时间复杂度与最内层循环主体的loop执行次数有关， 与有多少重循环无关，O(n) 
## space complexity
>> 只需要分配2个指针的额外内存，所以space 是O(1)

## 几种类型的双指针及相关题目
1. 同向：特点是指针不回头，全0 子串数量 - slow，fast， 基本等价于sliding window 
2. 相向：two sum， three sum， left， right
3. 背向：最长回文子串

## template 
### 相向双指针 - patition in quicksort 
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
### 背向双指针 
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
### 同向双指针 - 快慢指针 

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

## 高频题目整理

### 相向双指针
- #1. Two Sum 
    > 暴力double for loop -> hashtable -> 排序双指针(如何排序 + index操作需要注意); 这里要求返回下面，如果返回值比较容易
- #167. Two Sum II - Input Array Is Sorted https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/ 
    > 排序之后返回index，比two sum 简单， 我觉得只有排序之后，才能相向双指针，否则没有意义
- #15. 3Sum https://leetcode.com/problems/3sum/
    > 最外层for loop 作为一个指针，内嵌while loop，考虑left，right指针, 这个难点在于去重，去重很多种办法， 包括set，还有左右移动，因为已经排序，所以用相邻位置的比较来去重
- #16. 3Sum Closest https://leetcode.com/problems/3sum-closest/ 
    > 比15简单，几个edge cases都考虑到了
- #259. 3Sum Smaller https://leetcode.com/problems/3sum-smaller/ 
    > res += right - left # 这步是关键，之前没有想清楚， 为什么是right-left，其实就是中间的都可以
- #18. 4Sum https://leetcode.com/problems/4sum/ 
    > 完全和3sum一样！ 就是复杂的一点, time o(n^3), space o(n) 
- #454. 4Sum II https://leetcode.com/problems/4sum-ii/ 
    > hashmap, time o(n^2), space o(n^2) 
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

### 新添加一些题
- #38. Count and Say https://leetcode.com/problems/count-and-say (用到拼接的思想，如何双指针)
- #30. Substring with Concatenation of All Words https://leetcode.com/problems/substring-with-concatenation-of-all-words (sliding window好题)
- #228. Summary Ranges https://leetcode.com/problems/summary-ranges/ 


## 题目答案和分析

### 1. Two Sum
```python
    nums = [(number, index) for index, number in enumerate(nums)]
    nums.sort()
    left, right = 0, len(nums) - 1
    while left < right:
        if nums[left][0] + nums[right][0] < target: # 小于target，left 右移
            left += 1
        elif nums[left][0] + nums[right][0] > target: # 大于target，right 左移
            right -= 1
        else: # 这个必须要有，就是说 == target，直接返回
            return sorted([nums[left][1], nums[right][1]]) # 是否拍需要看要求，这个需要从小到大，就排一下
    return 
```

### 15. 3Sum https://leetcode.com/problems/3sum/
```python
        if len(nums) < 3:
            return 
        
        n = len(nums)
        nums.sort()
        ans = []
        for i in range(n):
            left = i + 1
            right = n - 1
            if nums[i] > 0: #最小值大于0
                break
            if i >= 1 and nums[i] == nums[i-1]: # 差一点，i >=1 才行，这种去重的题还是挺不好弄的！
                continue # 如果相邻的重复，那么就move一下，去掉重复的，直接跳过这个loop
            # left, right = 0, n - 1 # 不是这样的，要和i相关
            while left < right:
                target = nums[i] + nums[left] + nums[right]
                if target < 0:
                    left += 1
                elif target > 0:
                    right -= 1
                else:
                    ans.append([nums[i], nums[left], nums[right]])
                    while left != right and nums[left] == nums[left + 1]: # 去重 left
                        left += 1
                    while left != right and nums[right] == nums[right - 1]: # 去重 right
                        right -= 1
                    left += 1
                    right -= 1
        return ans
```

### 18. 4Sum https://leetcode.com/problems/4sum/  
```python
        hashmap = {}
        for i in nums1:
            for j in nums2:
                if i + j in hashmap:
                    hashmap[i + j] += 1
                else:
                    hashmap[i + j] = 1
        res = 0
        for m in nums3:
            for n in nums4:
                if 0 - m - n in hashmap:
                    res += hashmap[0 - m - n]
                    
        return res
```




