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
# double pointer 
class Solution:
    def twoSum(self, nums, target):
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

# hashmap solution - better one！
class Solution:
    def twoSum(self, nums, target):
        if not nums: return 
        n = len(nums)
        hashmap = {}
        for i in range(n):
            residual = target - nums[i]
            if residual in hashmap:
                res = [hashmap[residual], i]
            hashmap[nums[i]] = i
        return sorted(res)
```

### 1099. Two Sum Less Than K https://leetcode.com/problems/two-sum-less-than-k/ 
Given an array nums of integers and integer k, return the maximum sum such that there exists i < j with nums[i] + nums[j] = sum and sum < k. If no i, j exist satisfying this equation, return -1.
``` 
Example 1:

Input: nums = [34,23,1,24,75,33,54,8], k = 60
Output: 58
Explanation: We can use 34 and 24 to sum 58 which is less than 60.
Example 2:

Input: nums = [10,20,30], k = 15
Output: -1
Explanation: In this case it is not possible to get a pair sum less that 15. 
```
> two sum 的一个变种!
```python
class Solution:
    def twoSumLessThanK(self, nums: List[int], k: int) -> int:
        # brute force 不对，草！    time o(n^2) and space o(1)     
        res = -1 # 这个最开始就应该等于-1
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)): # 不一样的数
                twosum = nums[i] + nums[j]
                if twosum < k:
                    res = max(res, twosum)
        return res    
        # double pointer - 这个思路就很清晰，time o(nlogn), space o(logn) to o(n)
        nums.sort()
        left, right = 0, len(nums) - 1
        res = - 1
        while left < right:
            twosum = nums[left] + nums[right]
            if twosum < k:
                left += 1
                res = max(res, twosum)
            else:
                right -= 1
        return res
```

### 167. Two Sum II - Input Array Is Sorted https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/ 
Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length. Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2. The tests are generated such that there is exactly one solution. You may not use the same element twice. Your solution must use only constant extra space.
```
Example 1:

Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].
Example 2:

Input: numbers = [2,3,4], target = 6
Output: [1,3]
Explanation: The sum of 2 and 4 is 6. Therefore index1 = 1, index2 = 3. We return [1, 3].
Example 3:

Input: numbers = [-1,0], target = -1
Output: [1,2]
Explanation: The sum of -1 and 0 is -1. Therefore index1 = 1, index2 = 2. We return [1, 2].
```
> two sum 变种
```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        # 就是two sum with double pointer but it is sorted 
        # 要求 o(1) space
        left, right = 0, len(numbers) - 1
        while left < right:
            if numbers[left] + numbers[right] < target:
                left += 1
            elif numbers[left] + numbers[right] > target:
                right -= 1
            else:
                return [left + 1, right + 1]
        return []
```

### 15. 3Sum https://leetcode.com/problems/3sum/
```python
class Solution:
    def threeSum(self, nums):
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
# hashmap solution
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
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

# double pointer solution
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        # 完全和3sum一样！ 就是复杂的一点, time o(n^3), space o(n)
        nums.sort()
        n = len(nums)
        res = []
        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]: continue
            for j in range(i+1, n):
                if j > i + 1 and nums[j] == nums[j - 1]: continue
                left = j + 1
                right = n - 1 
                while left < right:
                    sum_all = nums[i] + nums[j] + nums[left] + nums[right]
                    if sum_all < target:
                        left += 1
                    elif sum_all > target:
                        right -= 1
                    else:
                        res.append([nums[i], nums[j], nums[left], nums[right]])
                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                        
                        left += 1
                        right -= 1
            
        return res # 位置错了，在最外层，容易忽视！
```


### 187. Repeated DNA Sequences https://leetcode.com/problems/repeated-dna-sequences/

The DNA sequence is composed of a series of nucleotides abbreviated as 'A', 'C', 'G', and 'T'.

For example, "ACGAATTCCG" is a DNA sequence.
When studying DNA, it is useful to identify repeated sequences within the DNA.

Given a string s that represents a DNA sequence, return all the 10-letter-long sequences (substrings) that occur more than once in a DNA molecule. You may return the answer in any order.

Example 1:
```
Input: s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
Output: ["AAAAACCCCC","CCCCCAAAAA"]
```
Example 2:
```
Input: s = "AAAAAAAAAAAAA"
Output: ["AAAAAAAAAA"]
```
```python
# 不难，应该不算典型的sliding window！
class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        if not s: return 
        n = len(s)
        hashm = {}
        res = []
        for i in range(n - 10 + 1): # 别忘了+1
            curr = s[i:i + 10]
            hashm[curr] = hashm.get(curr, 0) + 1
            if hashm[curr] > 1:
                res.append(curr)
            
        return set(res) # 这个要加set 否则输出一样的
        # time o(n), space o(n)
```

### 209. Minimum Size Subarray Sum https://leetcode.com/problems/minimum-size-subarray-sum/ 

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
```python
# sliding window 模板题，不难，涉及subarray 
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        if sum(nums) < target: # 临界状态
           return 0
        n = len(nums)
        slow = 0
        res = inf
        sum_ = 0
        for fast in range(n):
            sum_ += nums[fast]
            while sum_ >= target:
                res = min(res, fast - slow + 1)
                sum_ -= nums[slow]
                slow += 1
        return res
```

---
✅✅✅✅  217, 219, 220 是连续三个contains duplicate 比较常见

### 217. Contains Duplicate https://leetcode.com/problems/contains-duplicate/ 
Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.
```
Example 1:
Input: nums = [1,2,3,1]
Output: true
Example 2:
Input: nums = [1,2,3,4]
Output: false
Example 3:
Input: nums = [1,1,1,3,3,4,3,2,4,2]
Output: true
```

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        # way 1 - hashtable
        record = set()
        for num in nums:
            if num not in record: # 注意是not in 不是is not in
                record.add(num)
            else:
                return True
        return False
        # time o(n), space o(n)
        
        # way 2 - sorting - optimal space - time o(nlogn), space o(1)
        nums.sort()
        print(nums)
        n = len(nums)
        for i in range(1, n):
            if nums[i-1] == nums[i]: # 要搞清楚题目含义
                return True
        return False
```


### 219. Contains Duplicate II https://leetcode.com/problems/contains-duplicate-ii/ 
Given an integer array nums and an integer k, return true if there are two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.

Example 1:
```
Input: nums = [1,2,3,1], k = 3
Output: true
```
Example 2:
```
Input: nums = [1,0,1,1], k = 1
Output: true
```
Example 3:
```
Input: nums = [1,2,3,1,2,3], k = 2
Output: false
```

```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        # brute force - 超时
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] == nums[j] and abs(i-j) <= k:
                    return True
        return False

        # hashtable 还是做出来了，之前方向不对！ 
        # 思路是，用hash找到所有相同的数，然后要注意更新，因为是一次遍历，如果之前的相等的数不work要update到当前值
        hashm = {}
        for i in range(len(nums)):
            if nums[i] not in hashm:
                hashm[nums[i]] = i
            else:
                dis = abs(i - hashm[nums[i]])
                if dis <= k:
                    return True
                else:
                    hashm[nums[i]] = i # 这步是关键，之前没有更新nums = [1,0,1,1], k = 1 就过不去
                    continue 
        return False
        time o(n), space o(n)
        
        # sliding window, 但不容易，容易顺序错了！
        s = set()
        for i, num in enumerate(nums):
            # 先判断是不是有违法的窗口
            if i > k:
                s.remove(nums[i - k - 1])
            # 如果没有违法的，看是否满足目标
            if num in s:
                return True
            # 上面都没有满足，加入set来判断， 这三条顺序都不能变！
            s.add(num)
        return False 
```


### 220. Contains Duplicate III  https://leetcode.com/problems/contains-duplicate-iii/

Given an integer array nums and two integers k and t, return true if there are two distinct indices i and j in the array such that abs(nums[i] - nums[j]) <= t and abs(i - j) <= k.
```
Example 1:
Input: nums = [1,2,3,1], k = 3, t = 0
Output: true
Example 2:
Input: nums = [1,0,1,1], k = 1, t = 2
Output: true
Example 3:
Input: nums = [1,5,9,1,5,9], k = 2, t = 3
Output: false
```
> 挺难的，不太了解，超出了通常sliding window的范围！在219的基础上可能做出来一些，但排序已经bucket 不好做

``` python
# brute force 过不了，意义不大，太easy了
# class Solution:
#     def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
        # # brute force  - o(n^2), space o (1)
        # n = len(nums)
        # for i in range(n):
        #     for j in range(i + 1, n):
        #         if abs(nums[i] - nums[j]) <= t and abs(i - j) <= k:
        #             return True
        # return False 
class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
        from sortedcontainers import SortedSet
        if not nums or t < 0: return False     # Handle special cases
        ss, n = SortedSet(), 0                 # Create SortedSet. `n` is the size of sortedset, max value of `n` is `k` from input
        for i, num in enumerate(nums):
            ceiling_idx = ss.bisect_left(num)  # index whose value is greater than or equal to `num`
            floor_idx = ceiling_idx - 1        # index whose value is smaller than `num`
            if ceiling_idx < n and abs(ss[ceiling_idx]-num) <= t: return True  # check right neighbour 
            if 0 <= floor_idx and abs(ss[floor_idx]-num) <= t: return True     # check left neighbour
            ss.add(num)
            n += 1
            if i - k >= 0:  # maintain the size of sortedset by finding & removing the earliest number in sortedset
                ss.remove(nums[i-k])
                n -= 1
        return False
```



--- 
✅  系列题，关于longest substring distinct characters 很多类似的题目， 总结一下！主要是hashtable，sliding window的结合，复杂的case需要dp. upstart 考了类似的题目！

✅✅ substring, subarray, subsequence 三种常见的问题，总结一下！ 


✅✅✅ Substring & string 类型的

### 159. Longest Substring with At Most Two Distinct Characters  https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters/

Given a string s, return the length of the longest substring that contains at most two distinct characters.
Example 1:
```
Input: s = "eceba"
Output: 3
Explanation: The substring is "ece" which its length is 3.
```
Example 2:
```
Input: s = "ccaabbb"
Output: 5
Explanation: The substring is "aabbb" which its length is 5.
```

```python
# 经典sliding window 模板题，
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        # 这个模板不错！
        slow = 0 
        n = len(s)
        # hashset = set() # 尽量用set来弄，或者hashmap
        hashmap = {}
        res = 0
        for fast in range(n):
            # 先去操作目标，放进hashmap
            hashmap[s[fast]] = hashmap.get(s[fast],0) + 1
            # 先判断是否满足条件，如果满足，操作
            if len(hashmap) <= 2:
                res = max(res, fast - slow + 1)
            # 不满足的话，想办法更新slow 指针
            while len(hashmap) > 2:
                head = s[slow]
                hashmap[head] -= 1
                if hashmap[head] == 0:
                    del hashmap[head]
                slow += 1
                
        return res
```

### 340. Longest Substring with At Most K Distinct Characters https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/ 
Given a string s and an integer k, return the length of the longest substring of s that contains at most k distinct characters.
```
Example 1:

Input: s = "eceba", k = 2
Output: 3
Explanation: The substring is "ece" with length 3.
Example 2:

Input: s = "aa", k = 1
Output: 2
Explanation: The substring is "aa" with length 2.
```
> 这个题如果难一点就是让返回所有的最长的substring这个要自己写一下

```python 
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        # 模板确实厉害！清晰
        slow, hashm = 0, {}
        res = 0
        for fast in range(len(s)):
            tail = s[fast]
            hashm[tail] = hashm.get(tail, 0) + 1
            if len(hashm) <= k:
                res = max(res, fast - slow + 1)
            while len(hashm) > k: #这个位置就是想清楚，不满足if 条件，用if还是while
                head = s[slow]
                hashm[head] -= 1
                if hashm[head] == 0:
                    del hashm[head]
                slow += 1
        return res
```

### 3. Longest Substring Without Repeating Characters https://leetcode.com/problems/longest-substring-without-repeating-characters/
Given a string s, find the length of the longest substring without repeating characters.
```
Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
Example 2:

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
```
> 这个和340非常一致，可以放到一起来做，一个follow up是返回所有的最长的substring！ upstart考了！

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 这个模板还是不错的！ 
        slow = 0
        hashmap = {}
        res = 0
        n = len(s)
        for fast in range(n):
            tail = s[fast]
            hashmap[tail] = hashmap.get(tail, 0) + 1
            if len(hashmap) == fast - slow + 1:
                res = max(res, fast - slow + 1)
            while fast - slow + 1 > len(hashmap):
                head = s[slow]
                hashmap[head] -= 1
                if hashmap[head] == 0:
                    del hashmap[head]
                slow += 1
        return res
```

> Follow-up：return all 最长的substring
Example
input = "fsfetwenwac"
output = ['sfetw', 'enwac'] 

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 这个模板还是不错的！ 
        slow = 0
        hashmap = {}
        max_len = 0
        n = len(s)
        res_list = [] 
        for fast in range(n):
            tail = s[fast]
            hashmap[tail] = hashmap.get(tail, 0) + 1
            if len(hashmap) == fast - slow + 1:
                # 这段是核心记录所有list的办法！其实不难
                print(fast - slow + 1, max_len)
                if fast - slow + 1 > max_len:
                    res_list = [] # 这个就是通过[]来不断update 
                    res_list.append((slow, fast))
                    print(res_list)
                    max_len = max(max_len, fast - slow + 1)
                elif fast - slow + 1 == max_len:
                    res_list.append((slow, fast))
            
            while fast - slow + 1 > len(hashmap):
                head = s[slow]
                hashmap[head] -= 1
                if hashmap[head] == 0:
                    del hashmap[head]
                slow += 1
        # 用tuple记录slow and fast 位置，然后最后一起输出
        output = []
        print(res_list)
        for i, j in res_list:
            output.append(s[i:j+1])
            
        return output
```

> 395 算是substring，但不算典型的sliding window 

### 395. Longest Substring with At Least K Repeating Characters https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/ 
Given a string s and an integer k, return the length of the longest substring of s such that the frequency of each character in this substring is greater than or equal to k.
```
Example 1:

Input: s = "aaabb", k = 3
Output: 3
Explanation: The longest substring is "aaa", as 'a' is repeated 3 times.
Example 2:

Input: s = "ababbc", k = 2
Output: 5
Explanation: The longest substring is "ababb", as 'a' is repeated 2 times and 'b' is repeated 3 times.
```
> 这个用brute force可以，但是sliding window很难写，关键是如何控制 window里面的最小值， 递归的方法不好想，不具有通用性！虽然看上去和340挺像，但实际上完全不一样！！！！

```python
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        # 这个题用正常的sliding window 很难做!
        # brute force 要想出来 double for loop - time o(n^2), space o(1)
        n = len(s)
        res = 0
        for i in range(n):
            for j in range(i + 1, n + 1): # 注意这个地方要n+1
                hashmap = Counter(s[i:j]) # 这个也要注意
                if min(hashmap.values()) >= k: # 这个是可以操作的
                    res = max(res, j - i)
        return res
            
# # 递归的解法有点秀！
# class Solution(object):
#     def longestSubstring(self, s, k):
#         if len(s) < k:
#             return 0
#         for c in set(s):
#             if s.count(c) < k:
#                 return max(self.longestSubstring(t, k) for t in s.split(c))
#         return len(s)
```

### 424. Longest Repeating Character Replacement https://leetcode.com/problems/longest-repeating-character-replacement/ 
You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times. Return the length of the longest substring containing the same letter you can get after performing the above operations.
```
Example 1:

Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace the two 'A's with two 'B's or vice versa.
Example 2:

Input: s = "AABABBA", k = 1
Output: 4
Explanation: Replace the one 'A' in the middle with 'B' and form "AABBBBA".
The substring "BBBB" has the longest repeating letters, which is 4.
```
> 不算substring，但是sliding windows相关，有一类题就是可以替换操作！
```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        # 应用模板不错的一个题
        slow, res, max_freq, hashm = 0, 0, 0, {}
        for fast in range(len(s)):
            tail = s[fast]
            hashm[tail] = hashm.get(tail, 0) + 1
            max_freq = max(max_freq, hashm[tail]) # 这是关键，统计frequency，和01问题的区别，当时给定1了
            if fast - slow + 1 <= max_freq + k: # 这步必须是<= 之前也有类似的问题
                res = max(res, fast - slow + 1)
            while fast - slow + 1 > max_freq + k:
                head = s[slow]
                hashm[head] -= 1
                if hashm[head] == 0:
                    del hashm[head]
                slow += 1
        return res 
```

### 438. Find All Anagrams in a String https://leetcode.com/problems/find-all-anagrams-in-a-string/ 
Given two strings s and p, return an array of all the start indices of p's anagrams in s. You may return the answer in any order.
An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
```
Example 1:

Input: s = "cbaebabacd", p = "abc"
Output: [0,6]
Explanation:
The substring with start index = 0 is "cba", which is an anagram of "abc".
The substring with start index = 6 is "bac", which is an anagram of "abc".
Example 2:

Input: s = "abab", p = "ab"
Output: [0,1,2]
Explanation:
The substring with start index = 0 is "ab", which is an anagram of "ab".
The substring with start index = 1 is "ba", which is an anagram of "ab".
The substring with start index = 2 is "ab", which is an anagram of "ab".
```
> sliding window 模板题
```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        # 模板确实不错！
        res = []
        slow = 0
        hash_s = {}
        hash_p = {}
        for char in p:
            hash_p[char] = hash_p.get(char, 0) + 1
        for fast in range(len(s)):
            hash_s[s[fast]] = hash_s.get(s[fast], 0) + 1
            if hash_s == hash_p:
                res.append(slow)
            if fast >= len(p) - 1:
                head = s[slow]
                hash_s[head] -= 1
                if hash_s[head] == 0: 
                    del hash_s[head]
                slow += 1
        return res
```

### 567. Permutation in String https://leetcode.com/problems/permutation-in-string/ 
Given two strings s1 and s2, return true if s2 contains a permutation of s1, or false otherwise. In other words, return true if one of s1's permutations is the substring of s2.
```
Example 1:

Input: s1 = "ab", s2 = "eidbaooo"
Output: true
Explanation: s2 contains one permutation of s1 ("ba").
Example 2:

Input: s1 = "ab", s2 = "eidboaoo"
Output: false
```
> 和438 很像主席细节
```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        # sliding window的题，和438非常像, 本身也算permutation
        hashs1 = {}
        hashs2 = {}
        for char in s1:
            hashs1[char] = hashs1.get(char, 0) + 1
        
        slow = 0
        for fast in range(len(s2)):
            hashs2[s2[fast]] = hashs2.get(s2[fast], 0) + 1
            if hashs2 == hashs1:
                return True
            if fast >= len(s1) - 1:
                head = s2[slow] # 注意细节，不是一味的背模板！
                hashs2[head] -= 1
                if hashs2[head] == 0:
                    del hashs2[head]
                slow +=1
        
        return False
```

### 1208. Get Equal Substrings Within Budget 
You are given two strings s and t of the same length and an integer maxCost. You want to change s to t. Changing the ith character of s to ith character of t costs |s[i] - t[i]| (i.e., the absolute difference between the ASCII values of the characters). Return the maximum length of a substring of s that can be changed to be the same as the corresponding substring of t with a cost less than or equal to maxCost. If there is no substring from s that can be changed to its corresponding substring from t, return 0.

Example 1:
Input: s = "abcd", t = "bcdf", maxCost = 3
Output: 3
Explanation: "abc" of s can change to "bcd".
That costs 3, so the maximum length is 3.

Example 2:
Input: s = "abcd", t = "cdef", maxCost = 3
Output: 1
Explanation: Each character in s costs 2 to change to character in t,  so the maximum length is 1.

Example 3:
Input: s = "abcd", t = "acde", maxCost = 0
Output: 1
Explanation: You cannot make any change, so the maximum length is 1.

``` python
class Solution:
    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        # 题没太明白，但似乎就是很straightforward
        cost, res = 0, 0
        slow = 0
        for fast in range(len(t)):
            cost += abs(ord(s[fast]) - ord(t[fast]))
            if cost <= maxCost:
                res = max(res, fast - slow + 1)
                
            while cost > maxCost:
                cost -= abs(ord(s[slow]) - ord(t[slow]))
                slow += 1
        
        return res

```


✅✅✅✅ 系列题，max consecutive ones 1,2,3 

### 485. Max Consecutive Ones https://leetcode.com/problems/max-consecutive-ones/ 

Given a binary array nums, return the maximum number of consecutive 1's in the array.
```
Example 1:

Input: nums = [1,1,0,1,1,1]
Output: 3
Explanation: The first two digits or the last three digits are consecutive 1s. The maximum number of consecutive 1s is 3.
Example 2:

Input: nums = [1,0,1,1,0,1]
Output: 2
``` 
```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        temp = 0 
        res = 0 
        for i in nums:
            if i == 1:
                temp += 1
            else:
                temp = 0
            res = max(res, temp)
            
        return res 
```


### 487. Max Consecutive Ones II  https://leetcode.com/problems/max-consecutive-ones-ii/ 
Given a binary array nums, return the maximum number of consecutive 1's in the array if you can flip at most one 0.
```
Example 1:

Input: nums = [1,0,1,1,0]
Output: 4
Explanation: Flip the first zero will get the maximum number of consecutive 1s. After flipping, the maximum number of consecutive 1s is 4.
Example 2:

Input: nums = [1,0,1,1,0,1]
Output: 4
```
> 比485复杂但基本也是模板

```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        # get(key) 方法在 key（键）不在字典中时，可以返回默认值 None 或者设置的默认值
        # dict[key] 在 key（键）不在字典中时，会触发 KeyError 异常。
        slow = 0
        num_zero = 0
        hashm = {}   
        res = 0
        for fast in range(len(nums)):
            tail = nums[fast]
            hashm[tail] = hashm.get(tail, 0) + 1
            if hashm.get(0, 0) <= 1:  # 如果直接call dict[key] 就会报错，因为没有0，可能
                res = max(res, fast - slow + 1)
            
            while hashm.get(0, 0) > 1: # 如果直接call dict[key] 就会报错，因为没有0，可能
                head = nums[slow]
                hashm[head] -= 1
                if hashm[head] == 0:
                    del hashm[head]
                
                slow += 1
        return res
```

### 1004. Max Consecutive Ones III  https://leetcode.com/problems/max-consecutive-ones-iii/ 

Given a binary array nums and an integer k, return the maximum number of consecutive 1's in the array if you can flip at most k 0's.
```
Example 1:

Input: nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2
Output: 6
Explanation: [1,1,1,0,0,1,1,1,1,1,1]
Bolded numbers were flipped from 0 to 1. The longest subarray is underlined.
Example 2:

Input: nums = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], k = 3
Output: 10
Explanation: [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1]
Bolded numbers were flipped from 0 to 1. The longest subarray is underlined.
```
> 和487 完全一样，从1变成k, 这个模板不错！

```python 
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        slow = 0
        num_zero = 0
        hashm = {}   
        res = 0
        for fast in range(len(nums)):
            tail = nums[fast]
            hashm[tail] = hashm.get(tail, 0) + 1
            if hashm.get(0, 0) <= k:  # 如果直接call dict[key] 就会报错，因为没有0，可能
                res = max(res, fast - slow + 1)
            
            while hashm.get(0, 0) > k: # 如果直接call dict[key] 就会报错，因为没有0，可能
                head = nums[slow]
                hashm[head] -= 1
                if hashm[head] == 0:
                    del hashm[head]
                
                slow += 1
        return res
```

### 1446. Consecutive Characters  https://leetcode.com/problems/consecutive-characters/ 
The power of the string is the maximum length of a non-empty substring that contains only one unique character.Given a string s, return the power of s.
```
Example 1:

Input: s = "leetcode"
Output: 2
Explanation: The substring "ee" is of length 2 with the character 'e' only.
Example 2:

Input: s = "abbcccddddeeeeedcba"
Output: 5
Explanation: The substring "eeeee" is of length 5 with the character 'e' only.
```
> one pass 这种相邻的题目是一类题，substring 的这个是最简单的！ 
```python
class Solution:
    def maxPower(self, s: str) -> int:
        res = 1 # 这个初始化是1 不是0
        maxtemp = 1
        for i in range(1, len(s)):
            if s[i] == s[i-1]:
                maxtemp += 1
                res = max(res, maxtemp)
            else:
                maxtemp = 1
        return res 
        # time o(n), space o(1)
```


### 2067. Number of Equal Count Substrings https://leetcode.com/problems/number-of-equal-count-substrings/
You are given a 0-indexed string s consisting of only lowercase English letters, and an integer count. A substring of s is said to be an equal count substring if, for each unique letter in the substring, it appears exactly count times in the substring. Return the number of equal count substrings in s. A substring is a contiguous non-empty sequence of characters within a string.
```
Example 1:

Input: s = "aaabcbbcc", count = 3
Output: 3
Explanation:
The substring that starts at index 0 and ends at index 2 is "aaa".
The letter 'a' in the substring appears exactly 3 times.
The substring that starts at index 3 and ends at index 8 is "bcbbcc".
The letters 'b' and 'c' in the substring appear exactly 3 times.
The substring that starts at index 0 and ends at index 8 is "aaabcbbcc".
The letters 'a', 'b', and 'c' in the substring appear exactly 3 times.
Example 2:

Input: s = "abcd", count = 2
Output: 0
Explanation:
The number of times each letter appears in s is less than count.
Therefore, no substrings in s are equal count substrings, so return 0.
```

```python
class Solution:
    def equalCountSubstrings(self, s: str, count: int) -> int:
        # aaabbcccb
        ans=0
        # The sizes of the substring windows are limited, namely, 1*count, 2*count ..... 26*count
        # window can have 1 unique_char, 2 unique_chars...... 26 unique_chars respectively.
        for al in range(1,27):
            ans+=self.count_fix_s(s,al*count,count)
        return ans
    def count_fix_s(self,s,w,count):
        # standard sliding window logic with fixed window size w with left and right being left index and right index respectively..
        left=0
        d={}
        ans=0
        for right in range(len(s)):
            d[s[right]]=d.get(s[right],0)+1
            if right-left+1>w:
                d[s[left]]-=1
                if d[s[left]]==0: del d[s[left]]
                left+=1
            
            if right-left+1 == w and set(d.values())==set([count]):
                ans+=1
        return ans
```




✅✅✅  Subarray 题型总结！
> 有一类就是和K结合，product less than K， summary less than K 

### 643. Maximum Average Subarray I  https://leetcode.com/problems/maximum-average-subarray-i/ 
You are given an integer array nums consisting of n elements, and an integer k. Find a contiguous subarray whose length is equal to k that has the maximum average value and return this value. Any answer with a calculation error less than 10-5 will be accepted.
```
Example 1:

Input: nums = [1,12,-5,-6,50,3], k = 4
Output: 12.75000
Explanation: Maximum average is (12 - 5 - 6 + 50) / 4 = 51 / 4 = 12.75
Example 2:

Input: nums = [5], k = 1
Output: 5.00000
```
> subarray 经典入门题
```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        slow = 0
        n = len(nums)
        sum_ = 0 
        res = -inf
        for fast in range(n):
            sum_ += nums[fast]
            # 不满足窗口条件，对slow操作， 有时候也用while？因为这个是fix 窗口，所以用if
            if fast >= k - 1:
                sum_ -= nums[slow]
                slow += 1
            # 这步可以理解！ 满足条件然后操作
            if fast - slow + 1 == k:
                res = max(res, sum_ / k)
        return res
```

### 644. Maximum Average Subarray II https://leetcode.com/problems/maximum-average-subarray-ii/ ❌❌❌ 
You are given an integer array nums consisting of n elements, and an integer k. Find a contiguous subarray whose length is greater than or equal to k that has the maximum average value and return this value. Any answer with a calculation error less than 10-5 will be accepted.
```
Example 1:

Input: nums = [1,12,-5,-6,50,3], k = 4
Output: 12.75000
Explanation:
- When the length is 4, averages are [0.5, 12.75, 10.5] and the maximum average is 12.75
- When the length is 5, averages are [10.4, 10.8] and the maximum average is 10.8
- When the length is 6, averages are [9.16667] and the maximum average is 9.16667
The maximum average is when we choose a subarray of length 4 (i.e., the sub array [12, -5, -6, 50]) which has the max average 12.75, so we return 12.75
Note that we do not consider the subarrays of length < 4.
Example 2:

Input: nums = [5], k = 1
Output: 5.00000
```
> 虽然是643相似，但这个题是二分法
```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        if not nums:
            return 0
        start, end = min(nums), max(nums)
        while end - start > 1e-5:
            mid = (start + end) / 2
            if self.check_subarray(nums, k, mid):
                start = mid
            else:
                end = mid
        return start
    def check_subarray(self, nums, k, average):
        prefix_sum = [0]
        for num in nums:
            prefix_sum.append(prefix_sum[-1] + num - average)

        min_prefix_sum = 0
        for i in range(k, len(nums) + 1):
            if prefix_sum[i] - min_prefix_sum >= 0:
                return True
            min_prefix_sum = min(min_prefix_sum, prefix_sum[i - k + 1])
        return False
```


### 713. Subarray Product Less Than K https://leetcode.com/problems/subarray-product-less-than-k/ 
Given an array of integers nums and an integer k, return the number of contiguous subarrays where the product of all the elements in the subarray is strictly less than k.
```
Example 1:

Input: nums = [10,5,2,6], k = 100
Output: 8
Explanation: The 8 subarrays that have product less than 100 are:
[10], [5], [2], [6], [10, 5], [5, 2], [2, 6], [5, 2, 6]
Note that [10, 5, 2] is not included as the product of 100 is not strictly less than k.
Example 2:

Input: nums = [1,2,3], k = 0
Output: 0
```
> 之前的模板要改，不能直接用，因为先判断的话，其实window并不合法，所以要在最后存结果
```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        # 这个题不错，能不能输出所有的pair 
        if k <= 1: return 0
        slow = 0
        prod = 1
        res = 0
        for fast in range(len(nums)):
            prod *= nums[fast]
            # if prod < k: 都是错误的，不能在这存结果， 跟之前的模板不同，这个window不合法！
            #     res += fast - slow + 1  # 放在这就错误了，没有更新slow
            while prod >= k:
                prod /= nums[slow]
                slow += 1
            res += fast - slow + 1 
        return res
        # time o(n), space o(1)
        
        # brute force # time o(n^2) and space o(1) 会超时
        # res = 0
        # for i in range(len(nums)):
        #     prod = 1 
        #     for j in range(i, len(nums)):
        #         prod *= nums[j]
        #         if prod < k:
        #             res += 1
        #         else:
        #             continue 
        # return res 
        # time o(n^2), space o(1)
```
Follow-up: 如何输出所有符合的subarray # 如果要输出所有的subarrays 相当于在nums[slow:fast] 这个window里的subset 所有合集? 不好做！ 


### 1248. Count Number of Nice Subarrays https://leetcode.com/problems/count-number-of-nice-subarrays/
Given an array of integers nums and an integer k. A continuous subarray is called nice if there are k odd numbers on it. Return the number of nice sub-arrays.
```
Example 1:

Input: nums = [1,1,2,1,1], k = 3
Output: 2
Explanation: The only sub-arrays with 3 odd numbers are [1,1,2,1] and [1,2,1,1].
Example 2:

Input: nums = [2,4,6], k = 1
Output: 0
Explanation: There is no odd numbers in the array.
Example 3:

Input: nums = [2,2,2,1,2,2,1,2,2,2], k = 2
Output: 16
```
> 这个题不错，控制窗口计算内部所有可能性
```python
class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        # 前缀和直接定义成奇数的个数和，而不是简单求和，根据问题需要！核心是理解，前缀和的定义！
        odd = 0
        res = 0 
        hashm = {0:1} # 这个初始条件是重要的
        for num in nums:
            if num % 2 == 1:
                odd += 1
            if odd >= k and odd - k in hashm:
                res += hashm[odd - k]
            hashm[odd] = hashm.get(odd, 0) + 1 # 需要用hashm来维护
            print(hashm)
        return res
    #   time o(n), space o(n)
    
        # sliding window 还没有解，感觉比较麻烦！
        right ,left = 0,0
        ans = 0 
        odd_cnt = 0
        cur_sub_cnt = 0
        for right in range(len(nums)):
            # 控制窗口
            if nums[right]%2 == 1:
                odd_cnt += 1
                cur_sub_cnt = 0
            # 判定条件
            while odd_cnt == k:
                if nums[left]%2 == 1: # 左边有odd
                    odd_cnt -= 1
                cur_sub_cnt += 1
                left += 1
                
            ans += cur_sub_cnt
        return ans 
    # time o(n), space o(1)
```


### 1695. Maximum Erasure Value
You are given an array of positive integers nums and want to erase a subarray containing unique elements. The score you get by erasing the subarray is equal to the sum of its elements. Return the maximum score you can get by erasing exactly one subarray. An array b is called to be a subarray of a if it forms a contiguous subsequence of a, that is, if it is equal to a[l],a[l+1],...,a[r] for some (l,r).
```
Example 1:

Input: nums = [4,2,4,5,6]
Output: 17
Explanation: The optimal subarray here is [2,4,5,6].
Example 2:

Input: nums = [5,2,1,2,5,2,1,2,5]
Output: 8
Explanation: The optimal subarray here is [5,2,1] or [1,2,5].
```
> 模板题
```python
class Solution:
    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        # 这种套模板和归纳的方法最有用了!
        n = len(nums)
        slow, hashmap = 0, {}
        sum_ = 0
        res = -inf 
        for fast in range(n):
            tail = nums[fast]
            sum_ += tail
            hashmap[tail] = hashmap.get(tail, 0) + 1
            if fast - slow + 1 == len(hashmap):
                res = max(res, sum_)
            while fast - slow + 1 > len(hashmap):
                head = nums[slow]
                hashmap[head] -= 1
                if hashmap[head] == 0:
                    del hashmap[head]
                sum_ -= nums[slow]
                slow += 1
        
        return res
```


### 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/ 

Given an array of integers nums and an integer limit, return the size of the longest non-empty subarray such that the absolute difference between any two elements of this subarray is less than or equal to limit.
```
Example 1:

Input: nums = [8,2,4,7], limit = 4
Output: 2 
Explanation: All subarrays are: 
[8] with maximum absolute diff |8-8| = 0 <= 4.
[8,2] with maximum absolute diff |8-2| = 6 > 4. 
[8,2,4] with maximum absolute diff |8-2| = 6 > 4.
[8,2,4,7] with maximum absolute diff |8-2| = 6 > 4.
[2] with maximum absolute diff |2-2| = 0 <= 4.
[2,4] with maximum absolute diff |2-4| = 2 <= 4.
[2,4,7] with maximum absolute diff |2-7| = 5 > 4.
[4] with maximum absolute diff |4-4| = 0 <= 4.
[4,7] with maximum absolute diff |4-7| = 3 <= 4.
[7] with maximum absolute diff |7-7| = 0 <= 4. 
Therefore, the size of the longest subarray is 2.
Example 2:

Input: nums = [10,1,2,4,7,2], limit = 5
Output: 4 
Explanation: The subarray [2,4,7,2] is the longest since the maximum absolute diff is |2-7| = 5 <= 5.
Example 3:

Input: nums = [4,2,2,2,4,4,2,2], limit = 0
Output: 3
```
> 了解一下sortedlist, sorteddict, sortedset 三个内置的函数from sortedcontainers import SortedList。 还可以用单调队列
https://leetcode.cn/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/solution/jue-dui-chai-bu-chao-guo-xian-zhi-de-zui-5bki/ 但是很麻烦！

```python
# class Solution:
#     def longestSubarray(self, nums: List[int], limit: int) -> int:
        
#         # brute force, time o(n^2), space o(1) 肯定是超时
#         res = 0
#         for i in range(len(nums)):
#             for j in range(i + 1, len(nums) + 1):
#                 subarray = nums[i:j]
#                 diff = abs(max(subarray) - min(subarray))
#                 if diff <= limit:
#                     res = max(res, len(subarray))
#         return res
                
        # # 这是一类题，对于subarray内部区间的操作，然后返回最长或者直接返回subarray 
        # # 这个sliding window可以！还是超时，这个max或者nums[slow:fast]这个太慢了！
        # slow = 0
        # res = 0
        # for fast in range(1, len(nums) + 1):
        #     subarray = nums[slow:fast]
        #     diff = abs(max(subarray) - min(subarray))
        #     if diff <= limit:
        #         res = max(res, len(subarray))
        #     else:
        #         slow += 1
        # return res 
    
        # # 这个sliding window可以！用了sort还是超时，这个太慢了
        # slow = 0
        # res = 0
        # for fast in range(1, len(nums) + 1):
        #     subarray = nums[slow:fast]
        #     subarray.sort() # nlogn
        #     diff = subarray[-1] - subarray[0]
        #     if diff <= limit:
        #         res = max(res, len(subarray))
        #     else:
        #         slow += 1
        # return res 
    
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        from sortedcontainers import SortedList
        s = SortedList()  # python内置，然后直接sorting
        n = len(nums)
        left = right = ret = 0

        while right < n:
            s.add(nums[right])
            while s[-1] - s[0] > limit:
                s.remove(nums[left])
                left += 1
            ret = max(ret, right - left + 1)
            right += 1
        
        return ret
    
# 时间复杂度：O(n \log n)O(nlogn)，其中 nn 是数组长度。向有序集合中添加或删除元素都是 O(\log n)O(logn) 的时间复杂度。每个元素最多被添加与删除一次。

# 空间复杂度：O(n)O(n)，其中 nn 是数组长度。最坏情况下有序集合将和原数组等大。

# Python Sorted Containers
# Sorted Containers is an Apache2 licensed sorted collections library, written in pure-Python, and fast as C-extensions.

# Python’s standard library is great until you need a sorted collections type. Many will attest that you can get really far without one, but the moment you really need a sorted list, sorted dict, or sorted set, you’re faced with a dozen different implementations, most using C-extensions without great documentation and benchmarking.

# In Python, we can do better. And we can do it in pure-Python!

# >>> from sortedcontainers import SortedList
# >>> sl = SortedList(['e', 'a', 'c', 'd', 'b'])
# >>> sl
# SortedList(['a', 'b', 'c', 'd', 'e'])
# >>> sl *= 10_000_000
# >>> sl.count('c')
# 10000000
# >>> sl[-3:]
# ['e', 'e', 'e']
# >>> from sortedcontainers import SortedDict
# >>> sd = SortedDict({'c': 3, 'a': 1, 'b': 2})
# >>> sd
# SortedDict({'a': 1, 'b': 2, 'c': 3})
# >>> sd.popitem(index=-1)
# ('c', 3)
# >>> from sortedcontainers import SortedSet
# >>> ss = SortedSet('abracadabra')
# >>> ss
# SortedSet(['a', 'b', 'c', 'd', 'r'])
# >>> ss.bisect_left('c')
# 2

``` 


### 2294. Partition Array Such That Maximum Difference Is K https://leetcode.com/problems/partition-array-such-that-maximum-difference-is-k/ 

You are given an integer array nums and an integer k. You may partition nums into one or more subsequences such that each element in nums appears in exactly one of the subsequences.

Return the minimum number of subsequences needed such that the difference between the maximum and minimum values in each subsequence is at most k.

A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.
```
Example 1:

Input: nums = [3,6,1,2,5], k = 2
Output: 2
Explanation:
We can partition nums into the two subsequences [3,1,2] and [6,5].
The difference between the maximum and minimum value in the first subsequence is 3 - 1 = 2.
The difference between the maximum and minimum value in the second subsequence is 6 - 5 = 1.
Since two subsequences were created, we return 2. It can be shown that 2 is the minimum number of subsequences needed.
Example 2:

Input: nums = [1,2,3], k = 1
Output: 2
Explanation:
We can partition nums into the two subsequences [1,2] and [3].
The difference between the maximum and minimum value in the first subsequence is 2 - 1 = 1.
The difference between the maximum and minimum value in the second subsequence is 3 - 3 = 0.
Since two subsequences were created, we return 2. Note that another optimal solution is to partition nums into the two subsequences [1] and [2,3].
Example 3:

Input: nums = [2,2,4,5], k = 0
Output: 3
Explanation:
We can partition nums into the three subsequences [2,2], [4], and [5].
The difference between the maximum and minimum value in the first subsequences is 2 - 2 = 0.
The difference between the maximum and minimum value in the second subsequences is 4 - 4 = 0.
The difference between the maximum and minimum value in the third subsequences is 5 - 5 = 0.
Since three subsequences were created, we return 3. It can be shown that 3 is the minimum number of subsequences needed.
```
> sorting + greedy 多想想还是不难的，但这种maximum 和minimum的可以归纳总结一下！

```python
class Solution:
    def partitionArray(self, nums: List[int], k: int) -> int:
        #看到sorting之后就想到了，就没往下再写写 time, O(nlogn), space o(1)
        nums.sort()
        result = 1
        last = nums[0]
        for num in nums:
            if num - last > k:
                last = num # 更新最后一个数，如果比他大，就要update，确实是greedy，之前向greedy 是否一定work
                result += 1
        return result
```




✅✅✅  Subsequence的类型题！ 很多要用DP？ 

### 674. Longest Continuous Increasing Subsequence https://leetcode.com/problems/longest-continuous-increasing-subsequence/
Given an unsorted array of integers nums, return the length of the longest continuous increasing subsequence (i.e. subarray). The subsequence must be strictly increasing. A continuous increasing subsequence is defined by two indices l and r (l < r) such that it is [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] and for each l <= i < r, nums[i] < nums[i + 1].
```
Example 1:

Input: nums = [1,3,5,4,7]
Output: 3
Explanation: The longest continuous increasing subsequence is [1,3,5] with length 3.
Even though [1,3,5,7] is an increasing subsequence, it is not continuous as elements 5 and 7 are separated by element
4.
Example 2:

Input: nums = [2,2,2,2,2]
Output: 1
Explanation: The longest continuous increasing subsequence is [2] with length 1. Note that it must be strictly
increasing.
```
> 这个题挺不错，很多种解法！
```python
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        
        # 这个题虽然是一个easy，但是很多种解法，很多不错的方法！
        # way 1 - 上来想到的是simulate 也可以叫greey？ 
        # 这里比较的是 i + 1 和 i, 然后从len(nums) - 1开始的
#         if len(nums) == 0: return 0
#         res = 1
#         count = 1
#         for i in range(len(nums) - 1):
#             if nums[i + 1] > nums[i]:
#                 count += 1
#             else:
#                 count = 1
#             res = max(res, count)
#         return res 
    
        # # 这个写法也可以，从1开始的，之前写的有问题
        # res = 1
        # count = 1
        # for i in range(1, len(nums)):
        #     if nums[i] > nums[i - 1]:
        #         count += 1
        #     else:
        #         count = 1
        #     res = max(res, count)
        # return res
        
        
        # way 2 - sliding window, double pointer 这里面sliding window 有2种可以做！
        # 方法1 是锚钉，像solution 说的那种
        # res, anchor = 0, 0 
        # for i in range(len(nums)):
        #     if i and nums[i - 1] >= nums[i]:
        #         anchor = i
        #     res = max(res, i - anchor + 1) # 在出现nums[i - 1] >= nums[i]之前，anchor 总是0， 没有更新，所以这个就是记录的最大值
        # return res
            
        
        # # 方法2 是滑动while 然后求最大长度 这个理解的不错！复习一下sliding windows！
        # if not nums: return 0
        # slow, fast = 0, 1
        # res = 1 
        # n = len(nums)
        # for fast in range(n):
        #     while fast < n and nums[fast] > nums[fast - 1]:
        #         fast += 1
        #     res = max(res, fast - slow)
        #     slow = fast 
        # return  res
        # # time 一样是 o(n) and space is o(1)
        
        # way 3 - DP 方法！
        # 确定dp的含义 dp[i] 以下标i为结尾的数组的连续递增子序列长度
        if len(nums) == 0: return 0
        dp = [1] * len(nums)
        res = 1
        for i in range(len(nums) - 1):
            if nums[i + 1] > nums[i]:
                dp[i + 1] = dp[i] + 1
            res = max(res, dp[i + 1])
        return res
```








✅✅✅ Swap 题型 Consecutive Ones， Group All 1's Together

### 1151. Minimum Swaps to Group All 1's Together
Given a binary array data, return the minimum number of swaps required to group all 1’s present in the array together in any place in the array.
```
Example 1:

Input: data = [1,0,1,0,1]
Output: 1
Explanation: There are 3 ways to group all 1's together:
[1,1,1,0,0] using 1 swap.
[0,1,1,1,0] using 2 swaps.
[0,0,1,1,1] using 1 swap.
The minimum is 1.
Example 2:

Input: data = [0,0,0,1,0]
Output: 0
Explanation: Since there is only one 1 in the array, no swaps are needed.
Example 3:

Input: data = [1,0,1,0,1,0,0,1,1,0,1]
Output: 3
Explanation: One possible solution that uses 3 swaps is [0,0,0,0,0,1,1,1,1,1,1].
```
> 转移把问题转化和拆解
```python
class Solution:
    def minSwaps(self, data: List[int]) -> int:
        # 这些题都是在于如何转化和理解，而不是单纯的思考，直接解肯定不行！
        # 这个意思就是1） 数出有多少个1， 2） 1的个数为窗口，其中0的最小个数
        sum_1 = 0
        for i in data:
            if i == 1:
                sum_1 += 1
        if sum_1 <= 1: return 0  # 注意这个边界条件，如果全是0，那么就是返回0
        slow = 0 
        res_min = inf 
        len_1 = sum_1 
        num_0 = 0
        for fast in range(len(data)):
            if data[fast] == 0:
                num_0 += 1
            if fast - slow + 1 == len_1: #这个判断条件要有！
                res_min = min(res_min, num_0)
            
            if fast >= len_1 - 1:
                if data[slow] == 0:
                    num_0 -= 1
                slow += 1
            
        return res_min
```

### 2134. Minimum Swaps to Group All 1's Together II
A swap is defined as taking two distinct positions in an array and swapping the values in them. A circular array is defined as an array where we consider the first element and the last element to be adjacent. Given a binary circular array nums, return the minimum number of swaps required to group all 1's present in the array together at any location.
```
Example 1:

Input: nums = [0,1,0,1,1,0,0]
Output: 1
Explanation: Here are a few of the ways to group all the 1's together:
[0,0,1,1,1,0,0] using 1 swap.
[0,1,1,1,0,0,0] using 1 swap.
[1,1,0,0,0,0,1] using 2 swaps (using the circular property of the array).
There is no way to group all 1's together with 0 swaps.
Thus, the minimum number of swaps required is 1.
Example 2:

Input: nums = [0,1,1,1,0,0,1,1,0]
Output: 2
Explanation: Here are a few of the ways to group all the 1's together:
[1,1,1,0,0,0,0,1,1] using 2 swaps (using the circular property of the array).
[1,1,1,1,1,0,0,0,0] using 2 swaps.
There is no way to group all 1's together with 0 or 1 swaps.
Thus, the minimum number of swaps required is 2.
Example 3:

Input: nums = [1,1,0,0,1]
Output: 0
Explanation: All the 1's are already grouped together due to the circular property of the array.
Thus, the minimum number of swaps required is 0.
```
> 和1151唯一区别是allow circle，这样就mod就可以! 思路一样是先数多少个1，然后最小0
```python
class Solution:
    def minSwaps(self, nums: List[int]) -> int:
        n = len(nums)
        cnt = sum(nums)
        if cnt == 0:
            return 0
        cur = 0
        for i in range(cnt):
            cur += (1 - nums[i])
        
        ans = cur
        for i in range(1, n):
            if nums[i - 1] == 0:
                cur -= 1
            if nums[(i + cnt - 1) % n] == 0:
                cur += 1
            ans = min(ans, cur)
        return ans

# 作者：LeetCode-Solution
# 链接：https://leetcode.cn/problems/minimum-swaps-to-group-all-1s-together-ii/solution/zui-shao-jiao-huan-ci-shu-lai-zu-he-suo-iaghf/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```
 
### 1703. Minimum Adjacent Swaps for K Consecutive Ones   ❌ hard
You are given an integer array, nums, and an integer k. nums comprises of only 0's and 1's. In one move, you can choose two adjacent indices and swap their values.
Return the minimum number of moves required so that nums has k consecutive 1's.
```
Example 1:

Input: nums = [1,0,0,1,0,1], k = 2
Output: 1
Explanation: In 1 move, nums could be [1,0,0,0,1,1] and have 2 consecutive 1's.
Example 2:

Input: nums = [1,0,0,0,0,0,1,1], k = 3
Output: 5
Explanation: In 5 moves, the leftmost 1 can be shifted right until nums = [0,0,0,0,0,1,1,1].
Example 3:

Input: nums = [1,1,0,1], k = 2
Output: 0
Explanation: nums already has 2 consecutive 1's.
```
> 太难，很麻烦！短时间搞不定





✅✅✅ Others 非高频

### 1052. Grumpy Bookstore Owner

There is a bookstore owner that has a store open for n minutes. Every minute, some number of customers enter the store. You are given an integer array customers of length n where customers[i] is the number of the customer that enters the store at the start of the ith minute and all those customers leave after the end of that minute. On some minutes, the bookstore owner is grumpy. You are given a binary array grumpy where grumpy[i] is 1 if the bookstore owner is grumpy during the ith minute, and is 0 otherwise. When the bookstore owner is grumpy, the customers of that minute are not satisfied, otherwise, they are satisfied. The bookstore owner knows a secret technique to keep themselves not grumpy for minutes consecutive minutes, but can only use it once.

Return the maximum number of customers that can be satisfied throughout the day.
```
Example 1:

Input: customers = [1,0,1,2,1,1,7,5], grumpy = [0,1,0,1,0,1,0,1], minutes = 3
Output: 16
Explanation: The bookstore owner keeps themselves not grumpy for the last 3 minutes. 
The maximum number of customers that can be satisfied = 1 + 1 + 1 + 1 + 7 + 5 = 16.
Example 2:

Input: customers = [1], grumpy = [0], minutes = 1
Output: 1
```
```python
class Solution:
    def maxSatisfied(self, customers: List[int], grumpy: List[int], minutes: int) -> int:
        # 非高频, 不太懂！
        sum_, max_sum, max_start = 0, 0, 0
        slow = 0
        for fast in range(len(customers)):
            if grumpy[fast] == 1:
                sum_ += customers[fast]
            
            if sum_ > max_sum:
                max_sum = sum_
                max_start = slow 
            
            if fast >= minutes - 1:
                if grumpy[slow]:
                    sum_ -= customers[slow]
                slow += 1
            
        for i in range(max_start, max_start + minutes):
            grumpy[i] = 0
        
        res = 0
        for i in range(len(customers)):
            if not grumpy[i]:
                res += customers[i]
            
        return res
```

### 1984. Minimum Difference Between Highest and Lowest of K Scores
You are given a 0-indexed integer array nums, where nums[i] represents the score of the ith student. You are also given an integer k. Pick the scores of any k students from the array so that the difference between the highest and the lowest of the k scores is minimized. Return the minimum possible difference.
```
Example 1:

Input: nums = [90], k = 1
Output: 0
Explanation: There is one way to pick score(s) of one student:
- [90]. The difference between the highest and lowest score is 90 - 90 = 0.
The minimum possible difference is 0.
Example 2:

Input: nums = [9,4,1,7], k = 2
Output: 2
Explanation: There are six ways to pick score(s) of two students:
- [9,4,1,7]. The difference between the highest and lowest score is 9 - 4 = 5.
- [9,4,1,7]. The difference between the highest and lowest score is 9 - 1 = 8.
- [9,4,1,7]. The difference between the highest and lowest score is 9 - 7 = 2.
- [9,4,1,7]. The difference between the highest and lowest score is 4 - 1 = 3.
- [9,4,1,7]. The difference between the highest and lowest score is 7 - 4 = 3.
- [9,4,1,7]. The difference between the highest and lowest score is 7 - 1 = 6.
The minimum possible difference is 2.
```
```python
class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        
        # 这种easy题必须得过，simulation，然后注意是k>2的case，还是排序和之前的greedy idea 类似
        if len(nums) == 1 or k == 1: # 注意这个边界条件
            return 0
        nums.sort() # nlogn
        res = inf 
        for i in range(len(nums) - (k - 1)): # 注意这种第k个，进来直接-1
            diff = nums[i + k - 1] - nums[i]
            res = min(res, diff)
        return res
    # time i(n-(k-1)), space o(1)
```


### 1423. Maximum Points You Can Obtain from Cards
There are several cards arranged in a row, and each card has an associated number of points. The points are given in the integer array cardPoints. In one step, you can take one card from the beginning or from the end of the row. You have to take exactly k cards. Your score is the sum of the points of the cards you have taken. Given the integer array cardPoints and the integer k, return the maximum score you can obtain.
```
Example 1:

Input: cardPoints = [1,2,3,4,5,6,1], k = 3
Output: 12
Explanation: After the first step, your score will always be 1. However, choosing the rightmost card first will maximize your total score. The optimal strategy is to take the three cards on the right, giving a final score of 1 + 6 + 5 = 12.
Example 2:

Input: cardPoints = [2,2,2], k = 2
Output: 4
Explanation: Regardless of which two cards you take, your score will always be 4.
Example 3:

Input: cardPoints = [9,7,7,9,7,7,9], k = 7
Output: 55
Explanation: You have to take all the cards. Your score is the sum of points of all cards.
```
```python
class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        # 思路不好想，需要是移除连续n-k的最小的 - 剩下的就是最大值
        # 这个题思路重要，然后这个fast指针的范围也重要！
        n = len(cardPoints) 
        if n == k:
            return sum(cardPoints)
        
        res_min = inf
        slow = 0
        sum_ = 0
        for fast in range(n):
            sum_ += cardPoints[fast]
            if fast >= n - k - 1: # 就是差在这 因为fast 从0开始，但是n-k就是从绝对开始
                res_min = min(res_min, sum_)
                sum_ -= cardPoints[slow]
                slow += 1
        
        return sum(cardPoints) - res_min
```

✅✅✅✅ 删除类型题，找到指定规则的数或者string，然后问如何删除，删除最小次数是多少

### 2091. Removing Minimum and Maximum From Array 
You are given a 0-indexed array of distinct integers nums. There is an element in nums that has the lowest value and an element that has the highest value. We call them the minimum and maximum respectively. Your goal is to remove both these elements from the array. A deletion is defined as either removing an element from the front of the array or removing an element from the back of the array. Return the minimum number of deletions it would take to remove both the minimum and maximum element from the array.
```
Example 1:

Input: nums = [2,10,7,5,4,1,8,6]
Output: 5
Explanation: 
The minimum element in the array is nums[5], which is 1.
The maximum element in the array is nums[1], which is 10.
We can remove both the minimum and maximum by removing 2 elements from the front and 3 elements from the back.
This results in 2 + 3 = 5 deletions, which is the minimum number possible.
Example 2:

Input: nums = [0,-4,19,1,8,-2,-3,5]
Output: 3
Explanation: 
The minimum element in the array is nums[1], which is -4.
The maximum element in the array is nums[2], which is 19.
We can remove both the minimum and maximum by removing 3 elements from the front.
This results in only 3 deletions, which is the minimum number possible.
```
> 纯simulation， 删除次数，找最少次数
```python
class Solution:
    def minimumDeletions(self, nums: List[int]) -> int:
        n = len(nums)
        maxvalue = max(nums)
        minvalue = min(nums)
        
        for index, num in enumerate(nums):
            if num == maxvalue:
                maxindex = index
            if num == minvalue:
                minindex = index
        
        left1 = maxindex + 1 
        right1 = n - maxindex
        left2 = minindex + 1
        right2 = n - minindex
        # 注意对称性，比如max,min可以换，4种case讨论
        case1 = left1 + right2
        case2 = left2 + right1
        case3 = max(left1, left2)
        case4 = max(right1, right2)
        return min(case1, case2, case3, case4)
```





❌❌❌  hard题 sliding window ❌❌❌ 
### 76. Minimum Window Substring   https://leetcode.com/problems/minimum-window-substring/
Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "". The testcases will be generated such that the answer is unique. A substring is a contiguous sequence of characters within the string.
```
Example 1:

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
Example 2:

Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.
Example 3:

Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.
``` 
> 很复杂，不会
```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        
        # write your code here
        target, source = t, s

        if len(target) == 0 or len(source) == 0:
            return ''

        m, n = len(target), len(source)
        target_c, sub_c = {}, {}

        for i in range(m):
            target_c[target[i]] = target_c.get(target[i], 0) + 1

        fast = 0 
        matched_chars = 0
        start, substring_len = 0, float('inf') 

        for slow in range(n):

            while fast < n and matched_chars < len(target_c):
                sub_c[source[fast]] = sub_c.get(source[fast], 0) + 1
                if sub_c[source[fast]] == target_c.get(source[fast], 0):
                    matched_chars += 1
                fast += 1

            if matched_chars == len(target_c):
                if substring_len > fast - slow:
                    substring_len = fast - slow 
                    start = slow 

            sub_c[source[slow]] -= 1
            if sub_c[source[slow]] == target_c.get(source[slow], 0) - 1:
                matched_chars -= 1

        if substring_len == float('inf') :
            return ''

        return source[start : start + substring_len]
```

### 239. Sliding Window Maximum https://leetcode.com/problems/sliding-window-maximum/ 
You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position. Return the max sliding window.

```
Example 1:
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
Example 2:
Input: nums = [1], k = 1
Output: [1]
```
> hard 不会，没有做出来，跳过
```python 
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        
        n = len(nums)
        q = collections.deque()
        for i in range(k):
            while q and nums[i] >= nums[q[-1]]:
                q.pop()
            q.append(i)

        ans = [nums[q[0]]]
        for i in range(k, n):
            while q and nums[i] >= nums[q[-1]]:
                q.pop()
            q.append(i)
            while q[0] <= i - k:
                q.popleft()
            ans.append(nums[q[0]])
        
        return ans
```
### 30. Substring with Concatenation of All Words https://leetcode.com/problems/substring-with-concatenation-of-all-words/ [hard]
You are given a string s and an array of strings words of the same length. Return all starting indices of substring(s) in s that is a concatenation of each word in words exactly once, in any order, and without any intervening characters. You can return the answer in any order.

Example 1:
```
Input: s = "barfoothefoobarman", words = ["foo","bar"]
Output: [0,9]
Explanation: Substrings starting at index 0 and 9 are "barfoo" and "foobar" respectively.
The output order does not matter, returning [9,0] is fine too.
```
Example 2:
```
Input: s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"]
Output: []
```
Example 3:
```
Input: s = "barfoofoobarthefoobarman", words = ["bar","foo","the"]
Output: [6,9,12]
```

```python
# 完全不会， 太难！ 
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        from collections import Counter
        if not s or not words:return []
        one_word = len(words[0])
        word_num = len(words)
        n = len(s)
        words = Counter(words)
        res = []
        for i in range(0, one_word):
            cur_cnt = 0
            left = i
            right = i
            cur_Counter = Counter()
            while right + one_word <= n:
                w = s[right:right + one_word]
                right += one_word
                cur_Counter[w] += 1
                cur_cnt += 1
                while cur_Counter[w] > words[w]:
                    left_w = s[left:left+one_word]
                    left += one_word
                    cur_Counter[left_w] -= 1
                    cur_cnt -= 1
                if cur_cnt == word_num :
                    res.append(left)
        return res
```




