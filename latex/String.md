# String 


## ord 字符串技巧 开26个字符串的数组

### 383. Ransom Note
Given two strings ransomNote and magazine, return true if ransomNote can be constructed by using the letters from magazine and false otherwise.

Each letter in magazine can only be used once in ransomNote.
```
Example 1:

Input: ransomNote = "a", magazine = "b"
Output: false
Example 2:

Input: ransomNote = "aa", magazine = "ab"
Output: false
Example 3:

Input: ransomNote = "aa", magazine = "aab"
Output: true
```
> string + hashmap 的技巧
```python
# 用数组做，时间O(n), 空间O(1)
        arr = [0] * 26
        for x in magazine:
            arr[ord(x) - ord('a')] += 1        
        for x in ransomNote:
            if arr[ord(x) - ord('a')] == 0:
                return False
            else:
                arr[ord(x) - ord('a')] -= 1
        return True
#因为题目所只有小写字母，那可以采用空间换取时间的哈希策略， 用一个长度为26的数组还记录magazine里字母出现的次数。
# 然后再用ransomNote去验证这个数组是否包含了ransomNote所需要的所有字母。
# 依然是数组在哈希法中的应用。
``` 

### 242. Valid Anagram 
Given two strings s and t, return true if t is an anagram of s, and false otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
```
Example 1:

Input: s = "anagram", t = "nagaram"
Output: true
Example 2:

Input: s = "rat", t = "car"
Output: false
``` 
>  这道题目和242.有效的字母异位词 (opens new window)很像，242.有效的字母异位词 (opens new window)相当于求 字符串a 和 字符串b 是否可以相互组成 ，而这道题目是求 字符串a能否组成字符串b，而不用管字符串b 能不能组成字符串a。本题判断第一个字符串ransom能不能由第二个字符串magazines里面的字符构成，但是这里需要注意两点。
```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        # build two hashmap
        hashs, hasht = {}, {}
        for char in s:
            hashs[char] = hashs.get(char, 0) + 1
        for char in t:
            hasht[char] = hasht.get(char, 0) + 1
        if hashs == hasht:
            return True
        else:
            return False
``` 




### 409. Longest Palindrome 
Given a string s which consists of lowercase or uppercase letters, return the length of the longest palindrome that can be built with those letters.

Letters are case sensitive, for example, "Aa" is not considered a palindrome here.
```
Example 1:

Input: s = "abccccdd"
Output: 7
Explanation: One longest palindrome that can be built is "dccaccd", whose length is 7.
Example 2:

Input: s = "a"
Output: 1
Explanation: The longest palindrome that can be built is "a", whose length is 1.
```
> 有点贪心，但是直接模拟，很多palindrome 难题

```python
class Solution:
    def longestPalindrome(self, s: str) -> int:
        from collections import Counter
        s2 = Counter(s)
        res = 0
        odd = 0
        flag_odd = False
        for val in s2.values():
            if val % 2 == 0:
                res += val
            else:
                res += val - 1
                flag_odd = True
        
        if flag_odd is True:
            return res + 1
        else:
            return res
        # time o(n), space o(1), greedy
```

### 266. Palindrome Permutation 
Given a string s, return true if a permutation of the string could form a palindrome.
```
Example 1:

Input: s = "code"
Output: false
Example 2:

Input: s = "aab"
Output: true
Example 3:

Input: s = "carerac"
Output: true
```

```python
class Solution:
    def canPermutePalindrome(self, s: str) -> bool:
        # greey strategy hashmap 都是2个，最多只有1个是1
        # 很多种方法这个题
        from collections import Counter
        hashm = Counter(s)
        res = 0 
        for i in hashm.keys():
            if hashm[i] % 2 == 1:
                res += 1
        return True if res <= 1 else False 
        # time o(n), space o(1)
``` 


### 2131. Longest Palindrome by Concatenating Two Letter Words 

You are given an array of strings words. Each element of words consists of two lowercase English letters.

Create the longest possible palindrome by selecting some elements from words and concatenating them in any order. Each element can be selected at most once.

Return the length of the longest palindrome that you can create. If it is impossible to create any palindrome, return 0.

A palindrome is a string that reads the same forward and backward.

```
Example 1:

Input: words = ["lc","cl","gg"]
Output: 6
Explanation: One longest palindrome is "lc" + "gg" + "cl" = "lcggcl", of length 6.
Note that "clgglc" is another longest palindrome that can be created.
Example 2:

Input: words = ["ab","ty","yt","lc","cl","ab"]
Output: 8
Explanation: One longest palindrome is "ty" + "lc" + "cl" + "yt" = "tylcclyt", of length 8.
Note that "lcyttycl" is another longest palindrome that can be created.
Example 3:

Input: words = ["cc","ll","xx"]
Output: 2
Explanation: One longest palindrome is "cc", of length 2.
Note that "ll" is another longest palindrome that can be created, and so is "xx".
```
> 各种分类讨论，容易忽略一些case，其实不难，但是推广到一般情况，比如每个element的长度比较长怎么办？
```python
class Solution:
    def longestPalindrome(self, words: List[str]) -> int:
        freq = Counter(words)   # 单词出现次数
        res = 0   # 最长回文串长度
        mid = False   # 是否含有中心单词
        for word, cnt in freq.items():
            # 遍历出现的单词，并更新长度
            rev = word[1] + word[0]   # 反转后的单词
            if word == rev:
                if cnt % 2 == 1:
                    mid = True
                res += 2 * (cnt // 2 * 2)
            elif word > rev:   # 避免重复遍历
                res += 4 * min(freq[word], freq[rev])
        if mid:
            # 含有中心单词，更新长度
            res += 2
        return res
    # time o(n), space o(n) 取决于hash的开销

# 作者：LeetCode-Solution
# 链接：https://leetcode.cn/problems/longest-palindrome-by-concatenating-two-letter-words/solution/lian-jie-liang-zi-mu-dan-ci-de-dao-de-zu-vs99/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
``` 

