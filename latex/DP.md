# DP


✅✅✅ Subarray DP ✅✅✅ 

### 152. Maximum Product Subarray https://leetcode.com/problems/maximum-product-subarray/ 
Given an integer array nums, find a contiguous non-empty subarray within the array that has the largest product, and return the product. The test cases are generated so that the answer will fit in a 32-bit integer. A subarray is a contiguous subsequence of the array.
```
Example 1:

Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
Example 2:

Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
```
> 经典DP 在subarray 类型的！ olk
```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        
        # dp 解法和之前不一样的
        n = len(nums)
        dpmax = [0] * n
        dpmin = [0] * n
        dpmax[0] = nums[0]
        dpmin[0] = nums[0]
        res = nums[0]
        
        for i in range(1, n):
            dpmax[i] = max(dpmax[i-1] * nums[i], nums[i], dpmin[i-1]*nums[i])
            dpmin[i] = min(dpmax[i-1] * nums[i], nums[i], dpmin[i-1]*nums[i])
            res = max(res, dpmax[i])
        
        return res
```

### 53. Maximum Subarray 
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum. A subarray is a contiguous part of an array.
```
Example 1:

Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
Example 2:

Input: nums = [1]
Output: 1
Example 3:

Input: nums = [5,4,-1,7,8]
Output: 23
``` 
> subarray 很多都是DP的， 有一部分可以double pointer
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # dp[i] 数组的含义是以nums[i]结尾的连续子数组的最大和
        # 关键是dp[i] 转换的含义，这里面要判断dp[i-1]的大小，如果>0, dp[i]会更大，但是dp[i-1]<0, 重新开始，因为nums[i]会更小如果加上dp[i-1] 
    
        n = len(nums)
        dp = [0] * n 
        dp[0] = nums[0]
        for i in range(1, n):
            if dp[i-1] > 0: #不是判断nums[i]的正负
                dp[i] = dp[i-1] + nums[i]
            else:
                dp[i] = nums[i]
        return max(dp) # 返回所有的最大值
        
        # brute force - 这是比较优化的方法，但超时
        res = -inf 
        n = len(nums)
        for i in range(n):
            curr = 0 
            for j in range(i, n):
                curr += nums[j]
                res = max(res, curr)
        
        return res
```


### 718. Maximum Length of Repeated Subarray
Given two integer arrays nums1 and nums2, return the maximum length of a subarray that appears in both arrays.
```
Example 1:

Input: nums1 = [1,2,3,2,1], nums2 = [3,2,1,4,7]
Output: 3
Explanation: The repeated subarray with maximum length is [3,2,1].
Example 2:

Input: nums1 = [0,0,0,0,0], nums2 = [0,0,0,0,0]
Output: 5
```

```python
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:

        # #  动态规划可以，和之前的maximum subsequence 一样的dp方程 
        # A = nums1
        # B = nums2 
        # n, m = len(A), len(B)
        # dp = [[0] * (m + 1) for _ in range(n + 1)]
        # ans = 0
        # for i in range(n - 1, -1, -1):
        #     for j in range(m - 1, -1, -1):
        #         dp[i][j] = dp[i + 1][j + 1] + 1 if A[i] == B[j] else 0
        #         ans = max(ans, dp[i][j])
        # return ans
    
        # time o(n x m) and space o(n x m)
        
        
        # way 2 - sliding window
        def maxLength(addA: int, addB: int, length: int) -> int:
            ret = k = 0
            for i in range(length):
                if A[addA + i] == B[addB + i]:
                    k += 1
                    ret = max(ret, k)
                else:
                    k = 0
            return ret
        
        A = nums1
        B = nums2 
        n, m = len(A), len(B)
        ret = 0
        for i in range(n):
            length = min(m, n - i)
            ret = max(ret, maxLength(i, 0, length))
        for i in range(m):
            length = min(n, m - i)
            ret = max(ret, maxLength(0, i, length))
        return ret

# 作者：LeetCode-Solution
# 链接：https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/solution/zui-chang-zhong-fu-zi-shu-zu-by-leetcode-solution/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```



✅✅✅ Stock price ✅✅✅ 


### 2110. Number of Smooth Descent Periods of a Stock
You are given an integer array prices representing the daily price history of a stock, where prices[i] is the stock price on the ith day.  A smooth descent period of a stock consists of one or more contiguous days such that the price on each day is lower than the price on the preceding day by exactly 1. The first day of the period is exempted from this rule. Return the number of smooth descent periods.

```
Example 1:

Input: prices = [3,2,1,4]
Output: 7
Explanation: There are 7 smooth descent periods:
[3], [2], [1], [4], [3,2], [2,1], and [3,2,1]
Note that a period with one day is a smooth descent period by the definition.
Example 2:

Input: prices = [8,6,7,7]
Output: 4
Explanation: There are 4 smooth descent periods: [8], [6], [7], and [7]
Note that [8,6] is not a smooth descent period as 8 - 6 ≠ 1.
Example 3:

Input: prices = [1]
Output: 1
Explanation: There is 1 smooth descent period: [1]
```

```python
class Solution:
    def getDescentPeriods(self, prices: List[int]) -> int:
        n = len(prices)
        res = 1   # 平滑下降阶段的总数，初值为 dp[0]
        prev = 1   # 上一个元素为结尾的平滑下降阶段的总数，初值为 dp[0]
        # 从 1 开始遍历数组，按照递推式更新 prev 以及总数 res
        for i in range(1, n):
            if prices[i] == prices[i-1] - 1:
                prev += 1
            else:
                prev = 1
            res += prev
        return res
    
    # time o(n) and space o(1) 算是非常简单的dp了，需要理解dp的定义和含义！
```



✅✅✅ Subsequence DP ✅✅✅ 


### 1048. Longest String Chain https://leetcode.com/problems/longest-string-chain/ 
> Google 高频， 06232022 

You are given an array of words where each word consists of lowercase English letters.

wordA is a predecessor of wordB if and only if we can insert exactly one letter anywhere in wordA without changing the order of the other characters to make it equal to wordB.

For example, "abc" is a predecessor of "abac", while "cba" is not a predecessor of "bcad".
A word chain is a sequence of words [word1, word2, ..., wordk] with k >= 1, where word1 is a predecessor of word2, word2 is a predecessor of word3, and so on. A single word is trivially a word chain with k == 1.

Return the length of the longest possible word chain with words chosen from the given list of words.
```
Example 1:

Input: words = ["a","b","ba","bca","bda","bdca"]
Output: 4
Explanation: One of the longest word chains is ["a","ba","bda","bdca"].
Example 2:

Input: words = ["xbc","pcxbcf","xb","cxbc","pcxbc"]
Output: 5
Explanation: All the words can be put in a word chain ["xb", "xbc", "cxbc", "pcxbc", "pcxbcf"].
Example 3:

Input: words = ["abcd","dbqca"]
Output: 1
Explanation: The trivial word chain ["abcd"] is one of the longest word chains.
["abcd","dbqca"] is not a valid word chain because the ordering of the letters is changed.
``` 
> 这个题挺好的，需要想到DP，还要想到 temp = word[:i] + word[i + 1:] 挺难的！ 
```python 

class Solution:
    def longestStrChain(self, words: List[str]) -> int:
        # 对words中的元素，按照每个元素的长度，从小到大进行排序
        words.sort(key=lambda word: len(word))
        # map集合
        dic = {}
        maxLen, minVal = 1, len(words[0])
        for word in words:
            cur = 1
            if len(word) > minVal:
                # 枚举一下，删除哪个字符，可以获得最长单词链
                for i in range(len(word)):
                    temp = word[:i] + word[i + 1:] # 每次check 当前word的所有可能性，比如删除某一个字符之后，是否在之前的dict里面！
                    # print(word, temp)
                    if temp in dic:
                        cur = max(cur, dic[temp] + 1)
            dic[word] = cur
            maxLen = max(maxLen, dic[word])
        return maxLen
        # time o(n), n is the length of the number of words; space o(n), due to create the dict? 

# 作者：bu-lao-er-huo
# 链接：https://leetcode.cn/problems/longest-string-chain/solution/javapythonmapdp-by-bu-lao-er-huo-o9sd/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。   
```



✅✅✅ Substring DP ✅✅✅  这几个都是hard


### 2272. Substring With Largest Variance https://leetcode.com/problems/substring-with-largest-variance/
The variance of a string is defined as the largest difference between the number of occurrences of any 2 characters present in the string. Note the two characters may or may not be the same. Given a string s consisting of lowercase English letters only, return the largest variance possible among all substrings of s.

A substring is a contiguous sequence of characters within a string.

```
Example 1:

Input: s = "aababbb"
Output: 3
Explanation:
All possible variances along with their respective substrings are listed below:
- Variance 0 for substrings "a", "aa", "ab", "abab", "aababb", "ba", "b", "bb", and "bbb".
- Variance 1 for substrings "aab", "aba", "abb", "aabab", "ababb", "aababbb", and "bab".
- Variance 2 for substrings "aaba", "ababbb", "abbb", and "babb".
- Variance 3 for substring "babbb".
Since the largest possible variance is 3, we return it.
Example 2:

Input: s = "abcde"
Output: 0
Explanation:
No letter occurs more than once in s, so the variance of every substring is 0.
```
> 类似 53 maximum subarry？ 比较难的DP https://leetcode.com/problems/maximum-subarray/ 
```python    
# class Solution:
#     def largestVariance(self, s: str) -> int:
#         ans = 0
#         for a, b in permutations(ascii_lowercase, 2):
#             diff, diff_with_b = 0, -inf
#             for ch in s:
#                 if ch == a:
#                     diff += 1
#                     diff_with_b += 1
#                 elif ch == b:
#                     diff -= 1
#                     diff_with_b = diff  # 记录包含 b 时的 diff
#                     if diff < 0:
#                         diff = 0
#                 if diff_with_b > ans:
#                     ans = diff_with_b
#         return ans
class Solution:
    def largestVariance(self, s: str) -> int:
        if s.count(s[0]) == len(s):
            return 0
        ans = 0
        diff = [[0] * 26 for _ in range(26)]
        diff_with_b = [[-inf] * 26 for _ in range(26)]
        for ch in s:
            ch = ord(ch) - ord('a')
            for i in range(26):
                if i == ch:
                    continue
                diff[ch][i] += 1  # a=ch, b=i
                diff_with_b[ch][i] += 1
                diff[i][ch] -= 1  # a=i, b=ch
                diff_with_b[i][ch] = diff[i][ch]
                if diff[i][ch] < 0:
                    diff[i][ch] = 0
                ans = max(ans, diff_with_b[ch][i], diff_with_b[i][ch])
        return ans

# 作者：endlesscheng
# 链接：https://leetcode.cn/problems/substring-with-largest-variance/solution/by-endlesscheng-5775/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```


### 2262. Total Appeal of A String https://leetcode.com/problems/total-appeal-of-a-string/ 

The appeal of a string is the number of distinct characters found in the string. For example, the appeal of "abbca" is 3 because it has 3 distinct characters: 'a', 'b', and 'c'. Given a string s, return the total appeal of all of its substrings.

A substring is a contiguous sequence of characters within a string.

```
Example 1:

Input: s = "abbca"
Output: 28
Explanation: The following are the substrings of "abbca":
- Substrings of length 1: "a", "b", "b", "c", "a" have an appeal of 1, 1, 1, 1, and 1 respectively. The sum is 5.
- Substrings of length 2: "ab", "bb", "bc", "ca" have an appeal of 2, 1, 2, and 2 respectively. The sum is 7.
- Substrings of length 3: "abb", "bbc", "bca" have an appeal of 2, 2, and 3 respectively. The sum is 7.
- Substrings of length 4: "abbc", "bbca" have an appeal of 3 and 3 respectively. The sum is 6.
- Substrings of length 5: "abbca" has an appeal of 3. The sum is 3.
The total sum is 5 + 7 + 7 + 6 + 3 = 28.
Example 2:

Input: s = "code"
Output: 20
Explanation: The following are the substrings of "code":
- Substrings of length 1: "c", "o", "d", "e" have an appeal of 1, 1, 1, and 1 respectively. The sum is 4.
- Substrings of length 2: "co", "od", "de" have an appeal of 2, 2, and 2 respectively. The sum is 6.
- Substrings of length 3: "cod", "ode" have an appeal of 3 and 3 respectively. The sum is 6.
- Substrings of length 4: "code" has an appeal of 4. The sum is 4.
The total sum is 4 + 6 + 6 + 4 = 20.
```
```python

class Solution:
    def appealSum(self, s: str) -> int:
        from collections import Counter
        # brute force, time o(n^2), space o(n) 超时
        res = 0
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                sub = s[i:j]
                num = Counter(sub)
                res += len(num)
        return res
    
    ## DP 
    def appealSum(self, s: str) -> int:
        l = len(s)
        dp = [0 for i in range(l + 1)]
        d = {}
        for i, n in enumerate(s):
            dp[i + 1] = dp[i] + i - d.get(n, -1) #当不存在j时,j为-1
            d[n] = i
        return sum(dp)

    # dp记录每个下标元素作为子序列尾部的总引力
    # 哈希表记录每个元素出现的最大下标
    # 当前下标为i,上一个元素下标为j时, 子序列起始下标在[0, j]之间时引力不变, [j + 1, i]之间时引力 + 1
    # 所以dp的转移方程 dp[i + 1] = dp[i] + i - j
    # 如此即可求解.

    ## 类似828的题，比较巧妙
    def appealSum(self, s: str) -> int:
        last = defaultdict(lambda: -1)
        res, n = 0, len(s)
        for i, c in enumerate(s):
            res += (i - last[c]) * (n - i)
            last[c] = i
        return res
    
# This solution is more like what we do for 828. Count Unique Characters of All Substrings of a Given String.
# You can take 828 as an another chanllendge to practice more.

# In a substring, multiple same character only get one point.
# We can consider that the first occurrence get the point.
# Now for each character, we count its countribution for all substring.

# For each character s[i],
# the substring must start before s[i] to contain s[i]
# and need to end after the last occurrence of s[i],
# otherwise the last occurrence of character s[i] will get the socre.

# In total, there are i - last[s[i]] possible start position,
# and n - i possible end position,
# so s[i] can contribute (i - last[s[i]]) * (n - i) points.

# From this formula, we can also the difference between problem 2262 and 828.

# Complexity: Time O(n), Space O(26)
```

 
### 828. Count Unique Characters of All Substrings of a Given String https://leetcode.com/problems/count-unique-characters-of-all-substrings-of-a-given-string/

Let's define a function countUniqueChars(s) that returns the number of unique characters on s.

For example, calling countUniqueChars(s) if s = "LEETCODE" then "L", "T", "C", "O", "D" are the unique characters since they appear only once in s, therefore countUniqueChars(s) = 5.
Given a string s, return the sum of countUniqueChars(t) where t is a substring of s.

Notice that some substrings can be repeated so in this case you have to count the repeated ones too.

```
Example 1:

Input: s = "ABC"
Output: 10
Explanation: All possible substrings are: "A","B","C","AB","BC" and "ABC".
Every substring is composed with only unique letters.
Sum of lengths of all substring is 1 + 1 + 1 + 2 + 2 + 3 = 10
Example 2:

Input: s = "ABA"
Output: 8
Explanation: The same as example 1, except countUniqueChars("ABA") = 1.
Example 3:

Input: s = "LEETCODE"
Output: 92
```
> 这个答案，总结规律 和2262 非常像， 经典公式 res += (i - j) * (j - k) 
https://leetcode.com/problems/count-unique-characters-of-all-substrings-of-a-given-string/discuss/128952/JavaC%2B%2BPython-One-pass-O(N) 
```python
    def uniqueLetterString(self, S):
        index = {c: [-1, -1] for c in ascii_uppercase}
        res = 0
        for i, c in enumerate(S):
            k, j = index[c]
            res += (i - j) * (j - k)
            index[c] = [j, i]
        for c in index:
            k, j = index[c]
            res += (len(S) - j) * (j - k)
        return res % (10**9 + 7)
```