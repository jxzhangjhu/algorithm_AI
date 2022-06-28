## 使用条件
1. 排序数组（30%-40% 是二分法）
2. 当面试官要求找一个比o(n)更小的时间复杂度算法的时候，99% 就是二分logn
3. 找到数组中的一个分割位置，使得左半部分满足某个条件，右半部分不满足 100% 就是二分
4. 找到一个最大、最小的值使得某个条件被满足 90% 

## time complexity
>> o(logn)
## space complexity
>> o(1)

## 几种类型的双指针及相关题目


### Problems
#### 显式二分
- #35. Search Insert Position https://leetcode.com/problems/search-insert-position/
    > 模板题

**多次二分** 
- #34. Find First and Last Position of Element in Sorted Array https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/ 
    > 找排序数组的左右边界, 直接用模板，二次二分法
- #33. Search in Rotated Sorted Array https://leetcode.com/problems/search-in-rotated-sorted-array/
    > 也是从分割点，应用二次二分法

**山峰+多次二分** 
- #852. Peak Index in a Mountain Arrayhttps://leetcode.com/problems/peak-index-in-a-mountain-array/
    > 山峰最easy题
- #1095. Find in Mountain Array https://leetcode.com/problems/find-in-mountain-array/ 
    > 是一个hard， 首先二分要在sorted 上做，否则不work，上来先排序，然后找到peak value 用一次二分，然后分别堆两部分做2次二分
- #162. Find Peak Element https://leetcode.com/problems/find-peak-element/
    > 山峰类型题，根据要求，判断相邻大小然后对start 和 end check
- #1901. Find a Peak Element II https://leetcode.com/problems/find-a-peak-element-ii/ 
    > 山峰改为2D矩阵，比较难相比之前

**math + API 熟悉二分思想**
- #367. Valid Perfect Square https://leetcode.com/problems/valid-perfect-square/
    > math 题，练习二分思想
- #278. First Bad Version https://leetcode.com/problems/first-bad-version/ 
    > call API 的题，这里面target不是number，而是true or false
- #374. Guess Number Higher or Lower https://leetcode.com/problems/guess-number-higher-or-lower/
    > API 题，类似278，经典二分思想 
- #633. Sum of Square Numbers https://leetcode.com/problems/sum-of-square-numbers/
    > math 题，开始以为二分，但其实双指针更容易

**矩阵+二分**
- #74. Search a 2D Matrix https://leetcode.com/problems/search-a-2d-matrix/ 
    > 矩阵题，类似graph，一直处理的不好，这个题要重新做， 用模板 0307 update
- #240. Search a 2D Matrix II  https://leetcode.com/problems/search-a-2d-matrix-ii/
    > 矩阵题，可以暴力干，不是特别典型的二分，需要重新做，用模板，0307 update


#### 隐式二分
**不太理解的几个二分**
- #540. Single Element in a Sorted Array https://leetcode.com/problems/single-element-in-a-sorted-array/ 
    > 仍然不是很清楚，如何确定二分的条件，这个题需要理解成，奇偶数判断，然后移动mid， 需要再做
- #644. Maximum Average Subarray II https://leetcode.com/problems/maximum-average-subarray-ii/ 
    > 是一个hard，用到了前缀和，没做出来，需要再做
- #528. Random Pick with Weight https://leetcode.com/problems/random-pick-with-weight/
    > 不理解题含义

**写判定函数，注意边界**
- #1300. Sum of Mutated Array Closest to Target https://leetcode.com/problems/sum-of-mutated-array-closest-to-target/ 
    > 一类题，需要写一个判定函数，然后注意边界，start, end = 0, max(arr) 否则出错
- #1060. Missing Element in Sorted Array https://leetcode.com/problems/missing-element-in-sorted-array/
    > 比1300 复杂，需要写missing function，不在边界内，如何选，处理边界问题， good题
- #1891. Cutting Ribbons https://leetcode.com/problems/cutting-ribbons/ 
    > 核心的难点是，能不能cut，我能想到左右边界1， max，但要定义一个cut函数:给定一个target，能不能cut用现在的ribbons
- #1011. Capacity To Ship Packages Within D Days https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/ 
    > 考判定函数的题，非常经典， 很多类似的题
- #875 Koko Eating Bananas  https://leetcode.com/problems/koko-eating-bananas/
    > 判定函数，和1011类似，一个模板可解决，注意边界是start, end = 1, max(piles)
- #1231. Divide Chocolate https://leetcode.com/problems/divide-chocolate/
    > 1011，875 同类型题，注意有一个边界case，需要单独处理
- #410. Split Array Largest Sum https://leetcode.com/problems/split-array-largest-sum/
    > 判定函数，start, end = max(nums), sum(nums) 注意细节，是个hard但不难

**二分混合其他**
- #2089. Find Target Indices After Sorting Array https://leetcode.com/problems/find-target-indices-after-sorting-array/
    > 多种解法，有一个统计解法，二分其实不是最优的
- #1062. Longest Repeating Substring https://leetcode.com/problems/longest-repeating-substring/
    > 哈希表+判断函数，这个是string，注意边界，start, end = 0, n



--- 


### template 
```python
class Solution:
    def binary_search(self, nums, target):
        # corner case 处理 - 这里等价于num is None or len(num) == 0
        if not nums: return -1 
        # 初始化
        start, end = 0, len(nums) - 1
        # 用start + 1 < end 而不是 start < end 的目的是为了避免死循环
        # the first position of target 的情况下不会出现死循环，但是在last position of target 的情况下会出现死循环
        # example, nums = [1, 1]， target = 1 
        while start + 1 < end:
            mid = start + (end - start) // 2 
            # or mid = (start + end) // 2
            # > = < 的逻辑先分开写，然后在看看 = 的情况是否能合并到其他分支里
            if nums[mid] < target: 
                start = mid 
            elif nums[mid] == target:
                end = mid 
            else: 
                end = mid 
            # 因为上面的循环退出条件是 start + 1 < end 
            # 因此这里循环结束的时候， start 和 end 的关系是相邻关系 (1和2，3和4 这种)
            # 因此需要再单独判断 start 和 end 这两个数谁是我们想要的答案
            # 如果是找 first position of target 就先看start， 否则就先看end 
            if nums[start] == target:
                return start 
            if nums[end] == target: 
                return end 
        return -1 

```

### Example - 162. Find Peak Element  (不用改code，可解852)
https://leetcode.com/problems/find-peak-element/ 

A peak element is an element that is strictly greater than its neighbors.

Given an integer array nums, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.

You may imagine that nums[-1] = nums[n] = -∞.

You must write an algorithm that runs in O(log n) time.


Example 1:
```
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.
```
Example 2:
```
Input: nums = [1,2,1,3,5,6,4]
Output: 5
Explanation: Your function can return either index number 1 where the peak element is 2, or index number 5 where the peak element is 6.
```

>Solution 
```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        
        if not nums:
            return -1
        
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            
            if nums[mid] > nums[mid + 1]:
                end = mid 
            else:
                start = mid 
               
        # 这步不在while 循环内
        if nums[start] > nums[end]:
            return start 
        else:
            return end
        
        # 这个题还可以改一下就是九章那个，not return index, instead the max number
        # return max(nums[start], nums[end])
            
        return -1 

    # time o(logn) and space o(1) 
```

### 1891 cut 木头的题，需要判定函数
```python
class Solution:
    def maxLength(self, ribbons: List[int], k: int) -> int:
        
        # 核心的难点是，能不能cut，我能想到左右边界1， max，但要定义一个cut函数
        # 这个函数就是说给定一个target，能不能cut用现在的ribbons
        
        def can_cut(target):
            count = 0 
            for num in ribbons:
                count += num // target    
            return count >= k 
        
        start, end = 1, max(ribbons)
        while start + 1 < end:
            mid = (start + end) // 2
            if can_cut(mid):
                start = mid 
            else:
                end = mid 
        
        if can_cut(end): return end
        if can_cut(start): return start
        
        return 0
```


### 1044. Longest Duplicate Substring https://leetcode.com/problems/longest-duplicate-substring/ 

Given a string s, consider all duplicated substrings: (contiguous) substrings of s that occur 2 or more times. The occurrences may overlap.

Return any duplicated substring that has the longest possible length. If s does not have a duplicated substring, the answer is "".
```
Example 1:

Input: s = "banana"
Output: "ana"
Example 2:

Input: s = "abcd"
Output: ""
```
> 难点不在二分，而是hash判断？ 如何写二分的check函数
```python
# class Solution:
    # def longestDupSubstring(self, s: str) -> str:
    #     # 很难，可以用binary search + 特殊算法
    #     # sliding window 
    #     ans=''
    #     max_len,start,end,n=0,0,1,len(s)
    #     # 每次看s[start:end]在之后s[start+1:]是否也出现，因为在s[start:]肯定出现一次，所以总共出现次数>=2
    #     # 出现的话，更新max_len,同时end后移，看更长的是否满足
    #     # 不出现的话，表明以start开头的子串不可能出现>=2次，start后移
    #     while end<n:
    #         if s[start:end] in s[start+1:]: # 判断子串出现至少两次
    #             # 如果子串长度超过当前最大值，更新最大值和ans
    #             if max_len<end-start:
    #                 max_len=end-start
    #                 ans=s[start:end]
    #             end+=1
    #             continue
    #         start+=1
    #     return ans

# 作者：ling520
# 链接：https://leetcode.cn/problems/longest-duplicate-substring/solution/jian-dan-hua-dong-hua-dong-chuang-kou-ji-hirr/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

class Solution:
    def longestDupSubstring(self, S: str) -> str:
        import functools
        A = [ord(c) - ord('a') for c in S]
        mod = 2**63 - 1
        n = len(S)
        def test(l):
            p = pow(26,l,mod)
            cur = functools.reduce(lambda x,y:(x*26+y)%mod,A[:l])
            seed = {cur}
            for index in range(l,n):
                cur =(cur * 26 + A[index] - A[index-l] * p) % mod
                if cur in seed:
                    return index - l + 1
                seed.add(cur)
            return -1
        low,high = 0,n
        res = 0
        while low < high:
            mid = (low + high + 1) // 2
            pos = test(mid)
            if pos != -1:
                low = mid
                res = pos
            else:
                high = mid - 1
        return S[res:res+low]

# 这道题使用后缀数组无法通过但使用二分法可以通过。具体的做法是：在[0,n)的区间内使用二分法寻找最长重复子串的长度，对于一个长度k，判断字符串中是否存在长度为k的子串的具体做法为：将每一个长度为k的字符串看成一个26进制的数，然后将其转化为十进制，由于结果可能很大，所以每次运算后都将结果对一个很大的数取模。然后判断这个数是否出现过，如果出现过，则说明存在长度为k的子串；如果没出现过，则将其存入一个set。由于每次都可以将前一个子串的第一个字符去掉然后加上后一个字符得到下一个26进制数，所以进制转换的开销很小。


# 作者：1033020837
# 链接：https://leetcode.cn/problems/longest-duplicate-substring/solution/er-fen-fa-pythonjie-fa-by-1033020837/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```


