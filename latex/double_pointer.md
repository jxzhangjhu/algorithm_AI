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

### 3. Longest Substring Without Repeating Characters https://leetcode.com/problems/longest-substring-without-repeating-characters/ 
```python
# 这个题upstart考了难一点的版本，比如要返回所有的substring怎么办？ 比如最长的所有的substring 
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

### 76. Minimum Window Substring https://leetcode.com/problems/minimum-window-substring/ [hard]
Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique. A substring is a contiguous sequence of characters within the string.
```
Example 1:

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
```
Example 2:
```
Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.
```
Example 3:
```
Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.
```

```python
# 经典sliding window，比较难，可能需要背一下？
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
✅  217, 219, 220 是连续三个contains duplicate 比较常见

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


--- 
✅  系列题，关于longest substring distinct characters 很多类似的题目， 总结一下！主要是hashtable，sliding window的结合，复杂的case需要dp 

✅  upstart 考了类似的题目！



