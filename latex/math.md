# Math 相关题型


400. Nth Digit  https://leetcode.com/problems/nth-digit/ 

Given an integer n, return the nth digit of the infinite integer sequence [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...].

Example 1:
```
Input: n = 3
Output: 3
```
Example 2:
```
Input: n = 11
Output: 0
Explanation: The 11th digit of the sequence 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... is a 0, which is part of the number 10.
```

``` python
class Solution:
    def findNthDigit(self, n: int) -> int:
        cur, base = 1, 9
        while n > cur * base:
            n -= cur * base
            cur += 1
            base *= 10
        print(cur,n)
        n -= 1
        # 数字
        num = 10 ** (cur - 1) + n // cur
        print(num)
        # 数字里的第几位
        idx = n % cur
        print(idx)
        return num // (10 ** (cur - 1 - idx)) % 10
```