
# Double pointer & sliding window 


# Recursion & backtracking 


# Graph (DFS/BFS)


# Tree


# Stack/Heap/Queue


# DP
## 问题特征 
1. 求最大或者最小
2. 如果求所有可能一般都需要backtracking

## 问题类型 - 1D，2D
1. 区间DP
2. 背包DP

## 求解思路
1. 确定DP数组以及下标的含义
2. 确定递推公式
3. 初始化DP数组
4. 确定遍历顺序
5. 举例推导dp数组

## Examples 

926. Flip String to Monotone Increasing 
https://leetcode.com/problems/flip-string-to-monotone-increasing/ 

> [2D DP数组类型] 求最小，符合DP的特征，同时2种情况讨论，确定用2d dp，关键在于如何确定dp数组的含义！
```python
class Solution: # dp 看题解思路 O(n) O(n)
    def minFlipsMonoIncr(self, S: str) -> int:
        # DP数组的含义才是最重要的
        # DP[i][0] 是到i这个位置的子串，结尾是0的，需要的最少次数
        # DP[i][1] 是i这个位置的子串，结尾是1的，需要的最少次数
        # 递归过程：s[i] = 0时有2种情况
        # 1. 对于dp[i][0]需要都是0, 所以不需要翻转dp[i][0] = dp[i-1][0]
        # 2. 对于dp[i][1]需要翻转，因此+1，这个时候是上个时刻最小值+1 dp[i][1] = min(dp[i-1][0], dp[i-1][1]) + 1
        # 当s[i] = 1时也有2种情况
        # 1. dp[i][0] 需要翻转 dp[i][0] = dp[i-1][0] + 1 也就是从1到0
        # 2. dp[i][1] 不需要翻转 dp[i][1] = min(dp[i - 1][0], dp[i - 1][1]) 都是0的时候翻转 
        # 特殊情况
        if not S: return 0
        n = len(S)
        # 递归初始化
        dp = [[0] * 2 for _ in range(n)]
        dp[0][0] = int(S[0] == '1')
        dp[0][1] = int(S[0] == '0')
        # 递归遍历从1到n
        for i in range(1, n):
            if S[i] == '0':
                dp[i][0] = dp[i - 1][0]
                dp[i][1] = min(dp[i - 1][0], dp[i - 1][1]) + 1
            else:
                dp[i][0] = dp[i - 1][0] + 1
                dp[i][1] = min(dp[i - 1][0], dp[i - 1][1])
        return min(dp[n - 1]) # 返回值，最大到n-1

class Solution: # dp 空间优化 O(n) O(1)
    def minFlipsMonoIncr(self, S: str) -> int:
        zero, one = 0, 0
        n = len(S)
        for i in range(n):
            if S[i] == '0':
                one = min(zero, one) + 1
            else:
                zero, one = zero + 1, min(zero, one)
        return min(one, zero)

class Solution: # 前缀和 看题解思路 O(n) O(n)
    def minFlipsMonoIncr(self, S: str) -> int:
        P = [0]
        n = len(S)
        for a in S:
            P.append(P[-1] + int(a))
        # return min(P[i] + n - i - (P[n] - P[i]) for i in range(n))
        return min(P[i] + n - i - (P[n] - P[i]) for i in range(n + 1))
        # return min(P[j] + len(S)-j-(P[-1]-P[j]) for j in range(len(P)))
```



# Linked list 


# Binary search 


# Array/String 


# Sorting 

# Prefix sum 





<!-- 






# Binary search 搜索 

### 隐式二分： 定义判定函数 + 二分模板，区间进行二分搜索
- #1231. Divide Chocolate https://leetcode.com/problems/divide-chocolate [hard]
- #1011. Capacity to Ship packages within D days https://leetcode.com/problems/capacity-to-ship-packages-within-d-days [hard]
- #410. Split Array Largest Sum https://leetcode.com/problems/split-array-largest-sum [hard]
- #719. Find K-th Smallest Pair Distance https://leetcode.com/problems/find-k-th-smallest-pair-distance/ [hard] 结合双指针和二分

### 矩阵搜索 2D matrix 
- #378. Kth Samllest Element in a Sorted Matrix https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix 
- #74. Search a 2D Matrix https://leetcode.com/problems/search-a-2d-matrix
- #240. Search a 2D Matrix II https://leetcode.com/problems/search-a-2d-matrix-ii 








---
# Double pointers and sliding windows 双指针和滑动窗口

### 相向双指针
- #1. Two Sum https://leetcode.com/problems/two-sum
- #15. 3Sum https://leetcode.com/problems/3sum
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

> 新添加一些题
- #38. Count and Say https://leetcode.com/problems/count-and-say (用到拼接的思想，如何双指针)
- #30. Substring with Concatenation of All Words https://leetcode.com/problems/substring-with-concatenation-of-all-words (sliding window好题)
- #228. Summary Ranges https://leetcode.com/problems/summary-ranges/ 








---
# Sorting 排序
python自带的sorted 排序算法是timsort
> best time - o(n), average and worst case is o(nlogn);  space o(n); 
### Quick sort - time o(nlogn), space o(1)
> 先整体有序，再局部有序，利用分治的思想，递归的程序设计方式
### Merge sort - time o(nlogn), space o(n)
> 先局部有序，再整体有序

- #148. Sort List https://leetcode.com/problems/sort-list/ 
    > 超级麻烦，要有merge sort，感觉这种排序题在linked list 难度很大，而且容易出
- #179. Largest Number https://leetcode.com/problems/largest-number/
    > 不理解，官方答案超玄乎，但还是高频很多地方考过！
- #75. Sort Colors https://leetcode.com/problems/sort-colors/ 
- #493. Reverse Pairs https://leetcode.com/problems/reverse-pairs/ 
- #23. Merge k Sorted Lists  https://leetcode.com/problems/merge-k-sorted-lists/ 










---
# Array and string 相关题

### In-place 要求O(1)处理的题
- #26. Remove Duplicates from Sorted Array https://leetcode.com/problems/remove-duplicates-from-sorted-array
- #283. Move Zeroes https://leetcode.com/problems/move-zeroes 
- #48. Rotate Image https://leetcode.com/problems/rotate-image [2D matrix]
- #189. Rotate Array https://leetcode.com/problems/rotate-array
- #41. First Missing Positive https://leetcode.com/problems/first-missing-positive [hard]

### 安排会议 
- #252. Meeting Rooms https://leetcode.com/problems/meeting-rooms/ 
- #253. Meeting Rooms II https://leetcode.com/problems/meeting-rooms-ii/ [heap也可以解]
- #1094. Car Pooling https://leetcode.com/problems/car-pooling/ [prefix sum 差分]
- #452. Minimum Number of Arrows to Burst Balloons https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/ 
- #1235. Maximum Profit in Job Scheduling https://leetcode.com/problems/maximum-profit-in-job-scheduling/ [hard]
- #2054. Two Best Non-Overlapping Events https://leetcode.com/problems/two-best-non-overlapping-events/ 

### 区间操作：插入，合并，删除，非overlap
- #56. Merge Intervals https://leetcode.com/problems/merge-intervals/ [前缀和prefix sum]
- #57. Insert Interval https://leetcode.com/problems/insert-interval/ 
- #1272. Remove Interval https://leetcode.com/problems/remove-interval/ 
- #435. Non-overlapping Intervals https://leetcode.com/problems/non-overlapping-intervals/ 

### Subsequence 很多题都没做过！
- #392. Is Subsequence https://leetcode.com/problems/is-subsequence 
- #792. Number of Matching Subsequences https://leetcode.com/problems/number-of-matching-subsequences 
- #727. Minimum Window Subsequence https://leetcode.com/problems/minimum-window-subsequence [hard]
- #300. Longest Increasing Subsequence https://leetcode.com/problems/longest-increasing-subsequence 

### Subarray/substring 连续的，也有很多题
- #1062. Longest Repeating Substring https://leetcode.com/problems/longest-repeating-substring
- #1044. Longest Duplicate Substring https://leetcode.com/problems/longest-duplicate-substring [hard]
- #5. Longest Palindromic Substring https://leetcode.com/problems/longest-palindromic-substring 
- #3. Longest Substring Without Repeating Characters https://leetcode.com/problems/longest-substring-without-repeating-characters

### Subset 
- #78. Subsets https://leetcode.com/problems/subsets 
- #90. Subsets II https://leetcode.com/problems/subsets-ii 
- #368. Largest Divisible Subset https://leetcode.com/problems/largest-divisible-subset 

### prefix sum 前缀和
> 基本题，前缀和差分
- #303. Range Sum Query - Immutable https://leetcode.com/problems/range-sum-query-immutable 
- #304. Range Sum Query 2D - Immutable https://leetcode.com/problems/range-sum-query-2d-immutable 

> subarry类型
- #53. Maximum Subarray https://leetcode.com/problems/maximum-subarray 
- #325. Maximum Size Subarray Sum Equals k https://leetcode.com/problems/maximum-size-subarray-sum-equals-k 
- #525. Contiguous Array  https://leetcode.com/problems/contiguous-array
- #560. Subarray Sum Equals K https://leetcode.com/problems/subarray-sum-equals-k 
- #1248. Count Number of Nice Subarrays https://leetcode.com/problems/count-number-of-nice-subarrays 

> key是前缀和mod k的余数
- #523. Continuous Subarray Sum https://leetcode.com/problems/continuous-subarray-sum
- #974. Subarray Sums Divisible by K https://leetcode.com/problems/subarray-sums-divisible-by-k
- #1590. Make Sum Divisible by P https://leetcode.com/problems/make-sum-divisible-by-p
- #1524. Number of Sub-arrays With Odd Sum https://leetcode.com/problems/number-of-sub-arrays-with-odd-sum 

>前缀积， 前缀后缀信息同时使用
- #152. Maximum Product Subarray https://leetcode.com/problems/maximum-product-subarray
- #238. Product of Array Except Self https://leetcode.com/problems/product-of-array-except-self
- #724. Find Pivot Index https://leetcode.com/problems/find-pivot-index 

>差分方法，很好用
- #370. Range Addition https://leetcode.com/problems/range-addition 
- #1109. Corporate Flight Bookings https://leetcode.com/problems/corporate-flight-bookings 
- #1854. Maximum Population Year https://leetcode.com/problems/maximum-population-year















---
# Linked list 链表

### 快慢双指针操作（detect cycle, get middle, get kth element）
- #141. Linked List Cycle https://leetcode.com/problems/linked-list-cycle
- #19. Remove Nth Node From End of List https://leetcode.com/problems/remove-nth-node-from-end-of-list 

### 翻转链表 reverse linked list (dummy head)
- #206. Reverse Linked List https://leetcode.com/problems/reverse-linked-list 
- #25. Reverse Nodes in k-Group https://leetcode.com/problems/reverse-nodes-in-k-group [hard]

### LRU/LFU
- #146. LRU Cache https://leetcode.com/problems/lru-cache 
- #460. LFU Cache https://leetcode.com/problems/lfu-cache [hard]

### Deep copy
- #138. Copy List with Random Pointer https://leetcode.com/problems/copy-list-with-random-pointer 

### Merge LinkedList 
- #2. Add Two Numbers https://leetcode.com/problems/add-two-numbers 










---
# Tree 树

### tree 的遍历traverse (in-order, pre-order, post-order)
- #314. Binary Tree Vertical Order Traversal https://leetcode.com/problems/binary-tree-vertical-order-traversal 
- #199. Binary Tree Right Side View https://leetcode.com/problems/binary-tree-right-side-view

### 递归方法 
- #124. Binary Tree Maximum Path Sum https://leetcode.com/problems/binary-tree-maximum-path-sum [hard]
- #366. Find Leaves of Binary Tree https://leetcode.com/problems/find-leaves-of-binary-tree 

### Lowest Common Ancestor 系列
- #236. Lowest Common Ancestor of a Binary Tree https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree
- #235. Lowest Common Ancestor of a Binary Search Tree https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree 
- #1650. Lowest Common Ancestor of a Binary Tree III https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii
- #1644. Lowest Common Ancestor of a Binary Tree II https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-ii
- #1123. Lowest Common Ancestor of Deepest Leaves https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves
- #1676. Lowest Common Ancestor of a Binary Tree IV https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iv 

### Binary Search Tree (BST 性质，中序遍历increasing)
- #98. Validate Binary Search Tree https://leetcode.com/problems/validate-binary-search-tree 

### 编码解码 
- #449. Serialize and Deserialize BST  https://leetcode.com/problems/serialize-and-deserialize-bst/ 
- #297. Serialize and Deserialize Binary Tree  https://leetcode.com/problems/serialize-and-deserialize-binary-tree/ 

### 把tree变成graph
- #863. All Nodes Distance K in Binary Tree https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree 










---
# Stack，Queue and Heap 栈，队列 和堆

### Stack 
> 括号题 (括号补充题:20,22,32,301,678)
- #921. Minimum Add to Make Parentheses Valid https://leetcode.com/problems/minimum-add-to-make-parentheses-valid 
- #1249. Minimum Remove to Make Valid Parentheses https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses 

> 计算器系列题
- #227. Basic Calculator II https://leetcode.com/problems/basic-calculator-ii 
- #224. Basic Calculator https://leetcode.com/problems/basic-calculator [hard]
- #770. Basic Calculator IV https://leetcode.com/problems/basic-calculator-iv [hard]
- #772. Basic Calculator III https://leetcode.com/problems/basic-calculator-iii [hard]

> nested list iterator 系列
- #339. Nested List Weight Sum https://leetcode.com/problems/nested-list-weight-sum 
- #341. Flatten Nested List Iterator https://leetcode.com/problems/flatten-nested-list-iterator 
- #364. Nested List Weight Sum II https://leetcode.com/problems/nested-list-weight-sum-ii 

> others
- #394. Decode String https://leetcode.com/problems/decode-string 
- #726. Number of Atoms https://leetcode.com/problems/number-of-atoms [hard]

> 单调栈一般都是optimal solution，但首先用brute force！
- #496. Next Greater Element I  https://leetcode.com/problems/next-greater-element-i/ 
- #503. Next Greater Element II https://leetcode.com/problems/next-greater-element-ii/
- #556. Next Greater Element III https://leetcode.com/problems/next-greater-element-iii/ (不太好想！)
- #739. Daily Temperatures https://leetcode.com/problems/daily-temperatures/ 
- #901. Online Stock Span https://leetcode.com/problems/online-stock-span/
- #316. Remove Duplicate Letters https://leetcode.com/problems/remove-duplicate-letters/ 
- #402. Remove K Digits https://leetcode.com/problems/remove-k-digits/ 
- #581. Shortest Unsorted Continuous Subarray https://leetcode.com/problems/shortest-unsorted-continuous-subarray/ (可以用但不是必须用，sorting的解可以接受）
- #2104. Sum of Subarray Ranges https://leetcode.com/problems/sum-of-subarray-ranges/ (单调栈的解非常复杂，虽然能优化！)
- #1762. Buildings With an Ocean View https://leetcode.com/problems/buildings-with-an-ocean-view/ 

> 接雨水还有histogram专题
- #84. Largest Rectangle in Histogram https://leetcode.com/problems/largest-rectangle-in-histogram/ 
- #907. Sum of Subarray Minimums https://leetcode.com/problems/sum-of-subarray-minimums/ 
- #42. Trapping Rain Water https://leetcode.com/problems/trapping-rain-water/ 
- #11. Container With Most Water https://leetcode.com/problems/container-with-most-water/ 

> 4个类似的单调栈  https://leetcode-cn.com/problems/remove-duplicate-letters/solution/yi-zhao-chi-bian-li-kou-si-dao-ti-ma-ma-zai-ye-b-4/ 


### Queue (BFS相关提，单独的比较少)
- #239. Sliding Window Maximum https://leetcode.com/problems/sliding-window-maximum 
- #346. Moving Average from Data Stream https://leetcode.com/problems/moving-average-from-data-stream 

### Heap 很多难题可以用heap， 很多hard
> top k 问题
- #215. Kth Largest Element in an Array https://leetcode.com/problems/kth-largest-element-in-an-array
- #347. Top K Frequent Elements https://leetcode.com/problems/top-k-frequent-elements

> 中位数问题
- #295. Find Median from Data Stream https://leetcode.com/problems/find-median-from-data-stream
- #4. Median of Two Sorted Arrays https://leetcode.com/problems/median-of-two-sorted-arrays [hard]










---
# Graph 图

### Graph traverse - BFS,DFS 模板
> 基本模板，有没有基本题呢？

### 矩阵图， 4周neighbor相连 
> 0-1 islands 系列
- #200. Number of Islands https://leetcode.com/problems/number-of-islands
- #305. Number of Islands II https://leetcode.com/problems/number-of-islands-ii [hard]
- #694. Number of Distinct Islands https://leetcode.com/problems/number-of-distinct-islands 
- #711. Number of Distinct Islands II https://leetcode.com/problems/number-of-distinct-islands-ii [hard]
- #1254. Number of Closed Islands https://leetcode.com/problems/number-of-closed-islands

> world search 系列
- #79. Word Search https://leetcode.com/problems/word-search
- #212. Word Search II https://leetcode.com/problems/word-search-ii [hard]

> others
- #417. Pacific Atlantic Water Flow https://leetcode.com/problems/pacific-atlantic-water-flow 
- 690. Employee Importance https://leetcode.com/problems/employee-importance 

### Data(state)看成node,operation变成edge - 很多时候变成动态规划
- #127. Word Ladder https://leetcode.com/problems/word-ladder [hard]
- #126. Word Ladder II https://leetcode.com/problems/word-ladder-ii [hard]
- #1345. Jump Game IV https://leetcode.com/problems/jump-game-iv 

### 拓扑排序topological sort 
- #269. Alien Dictionary https://leetcode.com/problems/alien-dictionary [hard]
- #310. Minimum Height Trees https://leetcode.com/problems/minimum-height-trees
- #366. Find Leaves of Binary Tree https://leetcode.com/problems/find-leaves-of-binary-tree
- #444. Sequence Reconstruction https://leetcode.com/problems/sequence-reconstruction

> 排课系列
- #207. Course Schedule https://leetcode.com/problems/course-schedule
- #210. Course Schedule II https://leetcode.com/problems/course-schedule-ii 
- #630. Course Schedule III https://leetcode.com/problems/course-schedule-iii [hard]
- #1462. Course Schedule IV https://leetcode.com/problems/course-schedule-iv [hard]

### 图是否有cycle和二分染色
- #785. Is Graph Bipartite? https://leetcode.com/problems/is-graph-bipartite
- #1192. Critical Connections in a Network https://leetcode.com/problems/critical-connections-in-a-network 

### 最长最短路径 - BFS
- #994. Rotting Oranges https://leetcode.com/problems/rotting-oranges 
- #909. Snakes and Ladders https://leetcode.com/problems/snakes-and-ladders 
- #1091. Shortest Path in Binary Matrix https://leetcode.com/problems/shortest-path-in-binary-matrix 
- #1293. Shortest Path in a Grid with Obstacles Elimination https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination 


### Backtracking + DFS + memo 
- #526. Beautiful Arrangement https://leetcode.com/problems/beautiful-arrangement 
- #22. Generate Parentheses https://leetcode.com/problems/generate-parentheses 

### BFS + binary search 
- #1102. Path With Maximum Minimum Value https://leetcode.com/problems/path-with-maximum-minimum-value 
- #778. Swim in Rising Water https://leetcode.com/problems/swim-in-rising-water [hard]


### Word 系列 
- #139. Word Break https://leetcode.com/problems/word-break 
- #140. Word Break II https://leetcode.com/problems/word-break-ii 
- #290. Word Pattern https://leetcode.com/problems/word-pattern 
- #291. Word Pattern II https://leetcode.com/problems/word-pattern-ii 







---
# DFS + backtracking 

### 排列组合系列
- #46. Permutations https://leetcode.com/problems/permutations 
- #47. Permutations II https://leetcode.com/problems/permutations-ii/
- #31. Next Permutation https://leetcode.com/problems/next-permutation
- #77. Combinations https://leetcode.com/problems/combinations 
- #78. Subsets https://leetcode.com/problems/subsets/
- #90. Subsets IIhttps://leetcode.com/problems/subsets-ii/
- #39. Combination Sum https://leetcode.com/problems/combination-sum/
- #40. Combination Sum II https://leetcode.com/problems/combination-sum-ii/
- #60. Permutation Sequence https://leetcode.com/problems/permutation-sequence/
- #131. Palindrome Partitioning https://leetcode.com/problems/palindrome-partitioning/
- #267. Palindrome Permutation II https://leetcode.com/problems/palindrome-permutation-ii/ 








---
# DP 动态规划

### Jump Game 系列 


 -->



