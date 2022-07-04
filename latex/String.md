# String 


✅✅✅ in place operation - 可以模拟，可以用double pointer的几个题✅✅✅

### 283. Move Zeroes https://leetcode.com/problems/move-zeroes/ 

Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.

***Note that you must do this in-place without making a copy of the array***
```
Example 1:

Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]
Example 2:

Input: nums = [0]
Output: [0]
```
> 关键是in-place operation 一般只能是双指针
```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # 这种in-place的只能是双指针在移动
        n = len(nums)
        slow, fast = 0, 0
        for fast in range(n):
            if nums[fast] != 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1
        
        return nums
``` 

### 26. Remove Duplicates from Sorted Array https://leetcode.com/problems/remove-duplicates-from-sorted-array/ 
Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.
```
Example 1:

Input: nums = [1,1,2]
Output: 2, nums = [1,2,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 1 and 2 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
Example 2:

Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 0, 1, 2, 3, and 4 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
```

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        j = 1
        for i in range(1, len(nums)):
            if nums[i]!=nums[j-1]:
                nums[j] = nums[i]
                j += 1
        return j
```


### 27. Remove Element https://leetcode.com/problems/remove-element/ 
Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The relative order of the elements may be changed.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.
```
Example 1:

Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2,_,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 2.
It does not matter what you leave beyond the returned k (hence they are underscores).
Example 2:

Input: nums = [0,1,2,2,3,0,4,2], val = 2
Output: 5, nums = [0,1,4,0,3,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums containing 0, 0, 1, 3, and 4.
Note that the five elements can be returned in any order.
It does not matter what you leave beyond the returned k (hence they are underscores).
```
> 本质还是快慢指针
```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        # 这种就是同向，快慢指针，不能想移动操作，而是slow，fast, 之前思想不对！
        slow, fast = 0, 0
        n = len(nums)
        while fast < n:
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
            
            fast += 1 
        return slow 
``` 


### 80. Remove Duplicates from Sorted Array II https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/ 

Given an integer array nums sorted in non-decreasing order, remove some duplicates in-place such that each unique element appears at most twice. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

```
Example 1:

Input: nums = [1,1,1,2,2,3]
Output: 5, nums = [1,1,2,2,3,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 1, 1, 2, 2 and 3 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
Example 2:

Input: nums = [0,0,1,1,1,1,2,3,3]
Output: 7, nums = [0,0,1,1,2,3,3,_,_]
Explanation: Your function should return k = 7, with the first seven elements of nums being 0, 0, 1, 1, 2, 3 and 3 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
```

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        j = 2
        for i in range(2, len(nums)):
            if nums[i] != nums[j - 2]:
                nums[j] = nums[i]
                j += 1
        return j
```








✅✅✅ ord 字符串技巧 开26个字符串的数组✅✅✅ 

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


### 844. Backspace String Compare
Given two strings s and t, return true if they are equal when both are typed into empty text editors. '#' means a backspace character. Note that after backspacing an empty text, the text will continue empty.
``` 
Example 1:

Input: s = "ab#c", t = "ad#c"
Output: true
Explanation: Both s and t become "ac".
Example 2:

Input: s = "ab##", t = "c#d#"
Output: true
Explanation: Both s and t become "".
Example 3:

Input: s = "a#c", t = "b"
Output: false
Explanation: s becomes "c" while t becomes "b".
```
> 正常模拟题，但是所谓的double pointer，不好想，只是优化了space
```python
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        # brute force, time o(n + m), space o(n + m)
        snew = []
        for i in range(len(s)):
            if s[i] != '#':
                snew.append(s[i])
            elif snew:
                snew.pop()
                
        tnew = []
        for j in range(len(t)):
            if t[j] != '#':
                tnew.append(t[j])
            elif tnew:
                tnew.pop()

        if snew == tnew:
            return True
        else:
            return False
        
# 双指针不好想啊！ time o(m+n), space o(1)
class Solution(object):
    def backspaceCompare(self, S, T):
        def F(S):
            skip = 0
            for x in reversed(S):
                if x == '#':
                    skip += 1
                elif skip:
                    skip -= 1
                else:
                    yield x

        return all(x == y for x, y in itertools.izip_longest(F(S), F(T)))
``` 



### 14. Longest Common Prefix 

Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".
```
Example 1:

Input: strs = ["flower","flow","flight"]
Output: "fl"
Example 2:

Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
```

```python
横向搜索Horizontal scanning: O(S) , where S is the sum of all characters in all strings.

# 依次比较相邻两个字符串之间最长公共前缀
# 将每次比较得到的最长公共前缀与下个字符串进行比较，并更新公共前缀
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ''
        
        prefix = strs[0]
        count = len(strs)
        for i in range(1, count):
            prefix = self.lcp(prefix, strs[i])
            if not prefix:
                break
        return prefix

    def lcp(self, str1, str2):
        length = min(len(str1), len(str2))
        index = 0
        while index < length and str1[index] == str2[index]:
            index += 1
        return str1[:index]
buid in 0(MN)

class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        
        ##时间O(MN)
        if not strs: 
            return ''
        ##alphabetic order !    
        #["flower","flow","flight"]
        s1 = min(strs)  #flight
        s2 = max(strs) #flower

        for i in range(len(s1)):
            if s1[i] != s2[i]:
                return s1[:i]
        return s1
Trie 0(S)

class Solution(object):
    def longestCommonPrefix(self, strs):
        trie = Trie()
        for s in strs:
            trie.insert(s)
        return trie.search(strs[0])
    
#执行search的方法，如果字典树中某个节点额长度大于1，表示在此处遇到了分支，需要终止
#那么end有什么用？比如['a','ab']，如果只比较3的条件，结果就成了"ab"，所以需要同时判断是否在该节点出现了某个单词的终止情况。

class Trie:
    def __init__(self):
        self.root = dict()

    def insert(self, word):
        cur = self.root
        for w in word:
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
        cur['end'] = True

    def search(self, word):
        res = ''
        cur = self.root
        for w in word:
            if len(cur) > 1 or cur.get('end'):
                return res
            if w in cur:
                res += w
            cur= cur[w]
        return res
```
