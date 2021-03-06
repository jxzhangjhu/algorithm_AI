## 使用条件
1. 滑动窗口 - 90% 的概率
2. 时间复杂度要求 O(n) - 80%的概率
3. 要求原地操作，只可以交换使用，不能使用额外空间，所以空间复杂度O(1) - 80% 
4. 有子数组subarray， 子字符串substring的关键词 - 50%
5. 有回文问题 palindrome 关键词 - 50% 

## time complexity
>> 时间复杂度与最内层循环主体的loop执行次数有关， 与有多少重循环无关，O(n) 
## space complexity
>> 只需要分配2个指针的额外内存，所以space 是O(1)

## 几种类型的双指针及相关题目
1. 同向：特点是指针不回头，全0 子串数量 - slow，fast， 基本等价于sliding window 
2. 相向：two sum， three sum， left， right
3. 背向：最长回文子串





--- 
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





---
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






---
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
            while fast < length and sum(nums[slow:fast]) < target:  # 这部分不是总是o(n)，这个time 是o(2n)
                fast += 1
            if sum(nums[slow:fast]) >= target:
                res = min(fast - slow, res)  # 不断迭代
                
        return res
 ```

#### Example: lintcode 1375 · Substring With At Least K Distinct Characters
https://www.lintcode.com/problem/1375/description

Description: Given a string S with only lowercase characters. Return the number of substrings that contains at least k distinct characters.

Example
Example 1:
```
Input: S = "abcabcabca", k = 4
Output: 0
Explanation: There are only three distinct characters in the string.
```
Example 2:
```
Input: S = "abcabcabcabc", k = 3
Output: 55
Explanation: Any substring whose length is not smaller than 3 contains a, b, c.
    For example, there are 10 substrings whose length are 3, "abc", "bca", "cab" ... "abc"
    There are 9 substrings whose length are 4, "abca", "bcab", "cabc" ... "cabc"
    ...
    There is 1 substring whose length is 12, "abcabcabcabc"
    So the answer is 1 + 2 + ... + 10 = 55.
```

>Solution
```python
    def kDistinctCharacters(self, s, k):
        # Write your code here
        # 这个题要考虑几个点：
            # 1. 如何设置slow and fast 指针的位置
            # 2. 如何判断是否unique - hash 
            # 3. 如何计数，res = res + n - fast + 1
        if not s:
            return -1 
        fast = 0 
        length = len(s)
        res = 0 
        for slow in range(length - k + 1): # 介绍loop 次数，不用走到头
            fast = slow + k # j 也是从 slow + k 开始记录，否则没有一段k的，肯定不行
            while fast < length and len(set(s[slow:fast])) < k: 
                fast += 1
            if len(set(s[slow:fast])) >= k:
                res += length - fast + 1   # 计算res 的方法，要理解，fast后面的长度都可以
            
        return res 

        # 时间复杂度 o(n)， 空间复杂度o(len(s)), s中不同的字符串个数，因为开了hash set 
```

#### Example:  76. Minimum Window Substring
https://leetcode.com/problems/minimum-window-substring/

Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique.

A substring is a contiguous sequence of characters within the string.

Example 1:
```
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

>Solution
```python
        # write your code here

        ## leetcode 一个hard 题 - 仍然可以套模板，但是要注意几个点

        # 1. 需要结合hash map来做，要用dict 
        # 2. 注意判断条件，比之前多了一些，不是只要个数，而是返回string
        # 3. 用matched char来记录，这个思路可以想到， 但是写的时候比较吃力
        # 4. 实际上是三个指针，slow, fast and matched_char 

        target, source = t, s

        if len(target) == 0 or len(source) == 0:
            return ''

        m, n = len(target), len(source)
        target_c, sub_c = {}, {}

        for i in range(m):  # 提前做好hash map
            target_c[target[i]] = target_c.get(target[i], 0) + 1

        fast = 0 
        matched_chars = 0
        start, substring_len = 0, float('inf') 

        for slow in range(n):

            while fast < n and matched_chars < len(target_c): # 模板不成立条件
                sub_c[source[fast]] = sub_c.get(source[fast], 0) + 1 

                if sub_c[source[fast]] == target_c.get(source[fast], 0): #模板成立条件
                    matched_chars += 1
                fast += 1

            # 记录数据，更新最短子串
            if matched_chars == len(target_c):
                if substring_len > fast - slow:
                    substring_len = fast - slow 
                    start = slow  # 这一步很重要，记录start，后面要输出整个string

            # 这个是之前模板没有的，需要减去因为start在移动
            sub_c[source[slow]] -= 1
            if sub_c[source[slow]] == target_c.get(source[slow], 0) - 1:
                matched_chars -= 1 # 因为移动，所以要update 

        if substring_len == float('inf') :
            return ''

        return source[start : start + substring_len] # return也要care 
```









--- 

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

## Example 
https://leetcode.com/problems/heaters/

Winter is coming! During the contest, your first job is to design a standard heater with a fixed warm radius to warm all the houses.

Every house can be warmed, as long as the house is within the heater's warm radius range. 

Given the positions of houses and heaters on a horizontal line, return the minimum radius standard of heaters so that those heaters could cover all houses.

Notice that all the heaters follow your radius standard, and the warm radius will the same.

Example 1:
```
Input: houses = [1,2,3], heaters = [2]
Output: 1
Explanation: The only heater was placed in the position 2, and if we use the radius 1 standard, then all the houses can be warmed.
```
Example 2:
```
Input: houses = [1,2,3,4], heaters = [1,4]
Output: 1
Explanation: The two heater was placed in the position 1 and 4. We need to use radius 1 standard, then all the houses can be warmed.
```
Example 3:
```
Input: houses = [1,5], heaters = [2]
Output: 3
```

>Solution
```python
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        
#         # 二分法，模板今晚弄好！ 
#         # 二分插入位置需要数组有序
#         heaters.sort()
#         # 最近距离的最大值为最小的加热半径
#         heat_radius = 0
#         # 遍历房屋找到最近的加热器距离
#         for house in houses:
#             raius = self.get_minimum_radius(house, heaters)
#             heat_radius = max(heat_radius, raius)
        
#         return heat_radius
    
    
#     def get_minimum_radius(self, house, heaters):
#         left, right = 0, len(heaters) - 1
#         while left + 1 < right:
#             mid = left + (right - left) // 2
#             if heaters[mid] <= house:
#                 left = mid
#             else:
#                 right = mid
#         # 在left 和right 中找到答案
#         left_distance = abs(heaters[left] - house)
#         right_distance = abs(heaters[right] - house)
        
#         return min(left_distance, right_distance)
        
#     # 二分法复杂度分析
#     # 排序heaters, o(m*logm)
#     # 遍历每一个房屋hosue， o(n)
#     # 二分house 在heater 中的插入位置 o(logm)
#     # 总时间复杂度 o((n+m)*logm) 
    
    
    # 双数组型同向双指针
        houses.sort()
        heaters.sort()
        n, m = len(houses), len(heaters)
        i, j = 0, 0 
        heat_radius = 0 
        while i < n and j < m:
            now_radius = abs(heaters[j] - houses[i])
            next_radius = float('inf')
            if j < m - 1:
                next_radius = abs(heaters[j + 1] - houses[i])
            if now_radius < next_radius:
                heat_radius = max(heat_radius, now_radius)
                i += 1
            else:
                j += 1
        return heat_radius 

    # 两个数组最多被分别遍历一次o(n+m)
    # 数组需要排序o(n*logn + m*logm)
    # 总时间复杂度为 o(n*logn + m*logm)
    # 空间复杂度o(1)
```      




## 非典型双指针 - substring with group 

### Example: 696. Count Binary Substrings 
https://leetcode.com/problems/count-binary-substrings/

Give a binary string s, return the number of non-empty substrings that have the same number of 0's and 1's, and all the 0's and all the 1's in these substrings are grouped consecutively.

Substrings that occur multiple times are counted the number of times they occur.


Example 1:
```
Input: s = "00110011"
Output: 6
Explanation: There are 6 substrings that have equal number of consecutive 1's and 0's: "0011", "01", "1100", "10", "0011", and "01".
Notice that some of these substrings repeat and are counted the number of times they occur.
Also, "00110011" is not a valid substring because all the 0's (and 1's) are not grouped together.
```
Example 2:
```
Input: s = "10101"
Output: 4
Explanation: There are 4 substrings: "10", "01", "10", "01" that have equal number of consecutive 1's and 0's.
```

>Solution 
```python 
    # 也是看逻辑，如果想到了，就容易很多，分组的思想，并不是特别double pointer 
    # 这个题不easy，挺不好的其实！ 逻辑不是很straightforward！ 关于substring的处理！ 特别是这个group的处理，也不好像！ 
    
        groups = [1]
        for i in range(1, len(s)):
            if s[i-1] != s[i]:
                groups.append(1) # 不同的就新开一个分类
            else:
                groups[-1] += 1 # 如果相同的话，就累加，相当于stack 从末尾往里面进！这句话是核心
                
            # print(groups)
            
        ans = 0
        for i in range(1, len(groups)):
            ans += min(groups[i-1], groups[i])
        return ans
```
