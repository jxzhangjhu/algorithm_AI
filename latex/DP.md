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
> subarray 很多都是DP的， 有一部分可以double pointer - 这是所有subarray based DP最经典的，也是非常主要的，brute force 超时过不了
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


### 2193. Minimum Number of Moves to Make Palindrome  https://leetcode.com/problems/minimum-number-of-moves-to-make-palindrome/

You are given a string s consisting only of lowercase English letters.

In one move, you can select any two adjacent characters of s and swap them.

Return the minimum number of moves needed to make s a palindrome.

Note that the input will be generated such that s can always be converted to a palindrome.

```
Example 1:

Input: s = "aabb"
Output: 2
Explanation:
We can obtain two palindromes from s, "abba" and "baab". 
- We can obtain "abba" from s in 2 moves: "aabb" -> "abab" -> "abba".
- We can obtain "baab" from s in 2 moves: "aabb" -> "abab" -> "baab".
Thus, the minimum number of moves needed to make s a palindrome is 2.
Example 2:

Input: s = "letelt"
Output: 2
Explanation:
One of the palindromes we can obtain from s in 2 moves is "lettel".
One of the ways we can obtain it is "letelt" -> "letetl" -> "lettel".
Other palindromes such as "tleelt" can also be obtained in 2 moves.
It can be shown that it is not possible to obtain a palindrome in less than 2 moves.
```
> 题目标注是double pointer，greey但实际就是greey，非常困难，基本不会
```python
class Solution:               
# https://leetcode.com/problems/minimum-number-of-moves-to-make-palindrome/discuss/1822174/C%2B%2BPython-Short-Greedy-Solution

# 贪心，完全想不到！
    
    def minMovesToMakePalindrome(self, s):
        s = list(s)
        res = 0
        while s:
            i = s.index(s[-1])
            if i == len(s) - 1:
                res += i / 2
            else:
                res += i
                s.pop(i)
            s.pop()
        return int(res)
    
# Explanation
# Considering the first and the last char in final palindrome.
# If they are neither the first nor the last char in the initial string,
# you must waste some steps:
# Assume start with "...a....a.."
# ".a.......a." can be ealier completed thand "a.........a".

# Then compare the situation "a....b..a...b"
# It takes same number of steps to "ab.........ba" and "ba.........ab".
# So we can actually greedy move the characters to match string prefix.

# Other reference: https://www.codechef.com/problems/ENCD12

# Complexity
# Time O(n^2), can be improved to O(nlogn) by segment tree
# Space O(n)
```

### 1312. Minimum Insertion Steps to Make a String Palindrome https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/ 

Given a string s. In one step you can insert any character at any index of the string.

Return the minimum number of steps to make s palindrome.

A Palindrome String is one that reads the same backward as well as forward.

```
Example 1:

Input: s = "zzazz"
Output: 0
Explanation: The string "zzazz" is already palindrome we don't need any insertions.
Example 2:

Input: s = "mbadm"
Output: 2
Explanation: String can be "mbdadbm" or "mdbabdm".
Example 3:

Input: s = "leetcode"
Output: 5
Explanation: Inserting 5 characters the string becomes "leetcodocteel".
```

```python
class Solution:
    def minInsertions(self, s: str) -> int:
        n = len(s)
        dp = [[0] * (n + 1) for i in range(n + 1)]
        for i in range(n):
            for j in range(n):
                dp[i + 1][j + 1] = dp[i][j] + 1 if s[i] == s[-j-1] else max(dp[i][j + 1], dp[i + 1][j])
        return n - dp[n][n]
    
# Intuition
# Split the string s into to two parts,
# and we try to make them symmetrical by adding letters.

# The more common symmetrical subsequence they have,
# the less letters we need to add.

# Now we change the problem to find the length of longest common sequence.
# This is a typical dynamic problem.


# Explanation
# Step1.
# Initialize dp[n+1][n+1],
# wheredp[i][j] means the length of longest common sequence between
# i first letters in s1 and j first letters in s2.

# Step2.
# Find the the longest common sequence between s1 and s2,
# where s1 = s and s2 = reversed(s)

# Step3.
# return n - dp[n][n]


# Complexity
# Time O(N^2)
# Space O(N^2)

class Solution(object):
    def minInsertions(self, s):
        """
        :type s: str
        :rtype: int
        """
        #定义dp数组:dp[i][j],使得s[i...j]成为回文串所需要最少的插入次数
        length = len(s)
        dp = [[0]*length for i in range(length)]
       
        #反向遍历
        for i in range(length-2,-1,-1):
            for j in range(i+1,length):
                #print(i,j)
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1]
                else:
                    dp[i][j] = min(dp[i+1][j]+1,dp[i][j-1]+1)
                #print(dp[i][j])

        return dp[0][length-1]

# 步骤1：做选择，先将s[i-1,..j]或者是s[i..j+1]变回文串
# 步骤2将s[i..j]或者是s[i..j+1]变回文串
# 反向遍历

# 作者：a-bo-luo-zhi-zi
# 链接：https://leetcode.cn/problems/minimum-insertion-steps-to-make-a-string-palindrome/solution/yi-zui-xiao-cha-ru-ci-shu-gou-zao-hui-we-8vte/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```
 
### 1143. Longest Common Subsequence  https://leetcode.com/problems/longest-common-subsequence/

Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.

A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

For example, "ace" is a subsequence of "abcde".
A common subsequence of two strings is a subsequence that is common to both strings.

```
Example 1:

Input: text1 = "abcde", text2 = "ace" 
Output: 3  
Explanation: The longest common subsequence is "ace" and its length is 3.
Example 2:

Input: text1 = "abc", text2 = "abc"
Output: 3
Explanation: The longest common subsequence is "abc" and its length is 3.
Example 3:

Input: text1 = "abc", text2 = "def"
Output: 0
Explanation: There is no such common subsequence, so the result is 0.
``` 

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        
        # 核心还是状态转移方程，根据hint2
        # if text1[i] == text2[j], dp[i][j] = dp[i-1][j-1] + 1
        # otherwise, dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        n, m = len(text1), len(text2)
        dp = [[0] * (m + 1) for _ in range(n + 1)] # 2d matrix [n x m] 这个容易错！
        print(dp)
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            
        return dp[n][m]
``` 



## 经典DP

✅✅✅ Stock price ✅✅✅ 

### 121. Best Time to Buy and Sell Stock
You are given an array prices where prices[i] is the price of a given stock on the ith day. You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock. Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

```
Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
Example 2:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.
```

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        
        # # brute force 暴力遍历 time o(n^2) and space o(1) 超时了！
        # if not prices: return 0
        # res = -inf
        # for i in range(len(prices)):
        #     for j in range(i + 1, len(prices)):
        #         diff = prices[j] - prices[i]
        #         res = max(res, diff)
        # return res if res > 0 else 0 
        # 结果是对的，但是超时了，注意corner case，比如都是负数，要返回0
        
        # # greedy 贪心策略也可以 - time o(n), space o(1)
        # if not prices: return 0
        # low = inf
        # res = -inf 
        # for i in range(len(prices)):
        #     low = min(low, prices[i])
        #     res = max(res, prices[i] - low)
        # return res
        
        # DP - 维护一维动态数组 
        n = len(prices)
        if n == 0: return 0 # 边界条件
        dp = [0] * n
        minprice = prices[0] 

        for i in range(1, n):
            minprice = min(minprice, prices[i])
            dp[i] = max(dp[i - 1], prices[i] - minprice)

        return dp[-1]
```


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


✅✅✅ climbing stairs ✅✅✅ 

### 70. Climbing Stairs
You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
```
Example 1:

Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
Example 2:

Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
```
```python
class Solution:
    def climbStairs(self, n: int) -> int: 
# 一样的斐波那契
        if n < 4:
            return n 
        fn2 = 2
        fn3 = 3
        fn = 0
        for i in range(4, n + 1):
            fn = fn2 + fn3 
            fn2 = fn3
            fn3 = fn 
        return fn 
# time o(n), space o(1)
````

✅✅✅ Coin change 背包 ✅✅✅ 


### 322. Coin Change https://leetcode.com/problems/coin-change/

You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

```
Example 1:

Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
Example 2:

Input: coins = [2], amount = 3
Output: -1
Example 3:

Input: coins = [1], amount = 0
Output: 0
```
> 完全背包
```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
         # 记整数数组 coins 的长度为 nn。为便于状态更新，减少对边界的判断，初始二维 dpdp 数组维度为 {(n+1) \times (*)}(n+1)×(∗)，其中第一维为 n+1n+1 也意味着：第 ii 种硬币为 coins[i-1]coins[i−1]，第 11 种硬币为 coins[0]coins[0]，第 00 种硬币为空。
        n = len(coins)
        dp = [[amount+1] * (amount+1) for _ in range(n+1)]    # 初始化为一个较大的值，如 +inf 或 amount+1
        # 合法的初始化
        dp[0][0] = 0    # 其他 dp[0][j]均不合法
        # 完全背包：优化后的状态转移
        for i in range(1, n+1):             # 第一层循环：遍历硬币
            for j in range(amount+1):       # 第二层循环：遍历背包
                if j < coins[i-1]:          # 容量有限，无法选择第i种硬币
                    dp[i][j] = dp[i-1][j]
                else:                       # 可选择第i种硬币
                    dp[i][j] = min( dp[i-1][j], dp[i][j-coins[i-1]] + 1 )

        ans = dp[n][amount] 
        return ans if ans != amount+1 else -1
```


### 983. Minimum Cost For Tickets 
You have planned some train traveling one year in advance. The days of the year in which you will travel are given as an integer array days. Each day is an integer from 1 to 365.

Train tickets are sold in three different ways:

a 1-day pass is sold for costs[0] dollars,
a 7-day pass is sold for costs[1] dollars, and
a 30-day pass is sold for costs[2] dollars.
The passes allow that many days of consecutive travel.

For example, if we get a 7-day pass on day 2, then we can travel for 7 days: 2, 3, 4, 5, 6, 7, and 8.
Return the minimum number of dollars you need to travel every day in the given list of days.
```
Example 1:

Input: days = [1,4,6,7,8,20], costs = [2,7,15]
Output: 11
Explanation: For example, here is one way to buy passes that lets you travel your travel plan:
On day 1, you bought a 1-day pass for costs[0] = $2, which covered day 1.
On day 3, you bought a 7-day pass for costs[1] = $7, which covered days 3, 4, ..., 9.
On day 20, you bought a 1-day pass for costs[0] = $2, which covered day 20.
In total, you spent $11 and covered all the days of your travel.
Example 2:

Input: days = [1,2,3,4,5,6,7,8,9,10,30,31], costs = [2,7,15]
Output: 17
Explanation: For example, here is one way to buy passes that lets you travel your travel plan:
On day 1, you bought a 30-day pass for costs[2] = $15 which covered days 1, 2, ..., 30.
On day 31, you bought a 1-day pass for costs[0] = $2 which covered day 31.
In total, you spent $17 and covered all the days of your travel.
```

```python
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        dp = [0 for _ in range(days[-1] + 1)]  # dp数组，每个元素代表到当前天数最少钱数，为下标方便对应，多加一个 0 位置
        days_idx = 0  # 设定一个days指标，标记应该处理 days 数组中哪一个元素
        for i in range(1, len(dp)):
            if i != days[days_idx]:  # 若当前天数不是待处理天数，则其花费费用和前一天相同
                dp[i] = dp[i - 1]
            else:
                # 若 i 走到了待处理天数，则从三种方式中选一个最小的
                dp[i] = min(dp[max(0, i - 1)] + costs[0],
                            dp[max(0, i - 7)] + costs[1],
                            dp[max(0, i - 30)] + costs[2])
                days_idx += 1
        return dp[-1]  # 返回最后一天对应的费用即可
# time o(n), space o(n)
    
# 作者：LotusPanda
# 链接：https://leetcode-cn.com/problems/minimum-cost-for-tickets/solution/xiong-mao-shua-ti-python3-dong-tai-gui-hua-yi-do-2/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。 
``` 


✅✅✅ Paint house  ✅✅✅ 

### 256. Paint House 

There is a row of n houses, where each house can be painted one of three colors: red, blue, or green. The cost of painting each house with a certain color is different. You have to paint all the houses such that no two adjacent houses have the same color.

The cost of painting each house with a certain color is represented by an n x 3 cost matrix costs.

For example, costs[0][0] is the cost of painting house 0 with the color red; costs[1][2] is the cost of painting house 1 with color green, and so on...
Return the minimum cost to paint all houses.
```
Example 1:

Input: costs = [[17,2,17],[16,16,5],[14,3,19]]
Output: 10
Explanation: Paint house 0 into blue, paint house 1 into green, paint house 2 into blue.
Minimum cost: 2 + 5 + 3 = 10.
Example 2:

Input: costs = [[7,6,2]]
Output: 2
```

```python
class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        
        # 这个答案特别好，仔细看看！也可能多种情况考虑
        for n in reversed(range(len(costs) - 1)):
            # Total cost of painting nth house red.
            costs[n][0] += min(costs[n + 1][1], costs[n + 1][2])
            # Total cost of painting nth house green.
            costs[n][1] += min(costs[n + 1][0], costs[n + 1][2])
            # Total cost of painting nth house blue.
            costs[n][2] += min(costs[n + 1][0], costs[n + 1][1])

        if len(costs) == 0: return 0
        return min(costs[0]) # Return the minimum in the first row.
``` 


### 265. Paint House II 
There are a row of n houses, each house can be painted with one of the k colors. The cost of painting each house with a certain color is different. You have to paint all the houses such that no two adjacent houses have the same color.

The cost of painting each house with a certain color is represented by an n x k cost matrix costs.

For example, costs[0][0] is the cost of painting house 0 with color 0; costs[1][2] is the cost of painting house 1 with color 2, and so on...
Return the minimum cost to paint all houses.

```
Example 1:

Input: costs = [[1,5,3],[2,9,4]]
Output: 5
Explanation:
Paint house 0 into color 0, paint house 1 into color 2. Minimum cost: 1 + 4 = 5; 
Or paint house 0 into color 2, paint house 1 into color 0. Minimum cost: 3 + 2 = 5.
Example 2:

Input: costs = [[1,3],[2,4]]
Output: 5
``` 

```python
class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:   
        # extend the 3 colors to k colors # 各种最优解，可能都需要看？
        n = len(costs)
        if n == 0: return 0
        k = len(costs[0])
        for house in range(1, n):
            for color in range(k):
                best = math.inf
                for previous_color in range(k):
                    if color == previous_color: continue
                    best = min(best, costs[house - 1][previous_color])
                costs[house][color] += best

        return min(costs[-1])
    # time o(n*k^2), space o(1) or o(k)？
```