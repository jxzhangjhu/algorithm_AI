# Stack 

✅✅✅  stack 经典题 ✅✅✅ 

### 150. Evaluate Reverse Polish Notation 

Evaluate the value of an arithmetic expression in Reverse Polish Notation.

Valid operators are +, -, *, and /. Each operand may be an integer or another expression.
Note that division between two integers should truncate toward zero.
It is guaranteed that the given RPN expression is always valid. That means the expression would always evaluate to a result, and there will not be any division by zero operation. 
```
Example 1:

Input: tokens = ["2","1","+","3","*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9
Example 2:

Input: tokens = ["4","13","5","/","+"]
Output: 6
Explanation: (4 + (13 / 5)) = 6
Example 3:

Input: tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
Output: 22
Explanation: ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
```

```python
class Solution:
#     def evalRPN(self, tokens: List[str]) -> int:
#     # lambda 写法1
#     operations = {
#         "+": lambda a, b: a + b,
#         "-": lambda a, b: a - b,
#         "/": lambda a, b: int(a / b),
#         "*": lambda a, b: a * b
#     }
    
#     stack = []
#     for token in tokens:
#         if token in operations:
#             number_2 = stack.pop()
#             number_1 = stack.pop()
#             operation = operations[token]
#             stack.append(operation(number_1, number_2))
#         else:
#             stack.append(int(token))
#     return stack.pop()

    def evalRPN(self, tokens):
        stack = []
        for token in tokens:
            if token not in "+-/*":
                stack.append(int(token))
                continue
            number_2 = stack.pop()
            number_1 = stack.pop()
            result = 0
            if token == "+":
                result = number_1 + number_2
            elif token == "-":
                result = number_1 - number_2
            elif token == "*":
                result = number_1 * number_2
            else:
                result = int(number_1 / number_2)
            stack.append(result)
        return stack.pop()
``` 


### 227. Basic Calculator II 
Given a string s which represents an expression, evaluate this expression and return its value. 

The integer division should truncate toward zero.

You may assume that the given expression is always valid. All intermediate results will be in the range of [-231, 231 - 1].

Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as eval().

```
Example 1:

Input: s = "3+2*2"
Output: 7
Example 2:

Input: s = " 3/2 "
Output: 1
Example 3:

Input: s = " 3+5 / 2 "
Output: 5
```
> 其他几个calculator 都比较难，是hard就不考虑了

```python
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        n = len(s)
        presign = '+'
        operator = '+-*/'
        num = 0
        for i in range(n):
            if s[i] !=' ' and s[i].isdigit():
                num = num * 10 + ord(s[i]) - ord('0')
            if i == n - 1 or s[i] in operator:
                if presign == '+':
                    stack.append(num)
                elif presign == '-':
                    stack.append(-num)
                elif presign == '*':
                    stack.append(stack.pop() * num)
                else:
                    # stack.append(stack.pop() // num) # 这是错的
                    stack.append(int(stack.pop() / num)) # 要用这个
                presign = s[i]
                num = 0
        return sum(stack)
    
        # time o(n) and space o(n)
``` 



### 735. Asteroid Collision 
We are given an array asteroids of integers representing asteroids in a row.

For each asteroid, the absolute value represents its size, and the sign represents its direction (positive meaning right, negative meaning left). Each asteroid moves at the same speed.

Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller one will explode. If both are the same size, both will explode. Two asteroids moving in the same direction will never meet.
```
Example 1:

Input: asteroids = [5,10,-5]
Output: [5,10]
Explanation: The 10 and -5 collide resulting in 10. The 5 and 10 never collide.
Example 2:

Input: asteroids = [8,-8]
Output: []
Explanation: The 8 and -8 collide exploding each other.
Example 3:

Input: asteroids = [10,2,-5]
Output: [10]
Explanation: The 2 and -5 collide resulting in -5. The 10 and -5 collide resulting in 10.
```
> stack 经典题
```python 
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        # s, p = [], 0
        # while p < len(asteroids):
        #     if not s or s[-1] < 0 or asteroids[p] > 0:
        #         s.append(asteroids[p])
        #     elif s[-1] <= -asteroids[p]:
        #         if s.pop() < -asteroids[p]:
        #             continue
        #     p += 1
        # return s

# https://leetcode.com/problems/asteroid-collision/discuss/904475/Python-3-or-Stack-Simply-Clean-O(N)-or-Explanation
        # 这个答案写的好，就是模拟这个过程 用stack
        # 解决好什么时候进stack，什么时候操作 s[-1]是关键，就是栈顶？ 
        # case 1 - 
        s = []
        for a in asteroids:
            while s and s[-1] > 0 and a < 0: # 必须是while
                if s[-1] + a < 0: s.pop() # abs(left) < abs(right)，当前pop，再check
                elif s[-1] + a > 0: break    # 跳出while
                else: s.pop(); break # left = right 也会跳出
            else: s.append(a)  
        return s
    
    # while else 的使用 - 执行越换，如果不满足，跳出执行else
```



## mono stack 单调栈

✅✅✅ subarray 类型的 ✅✅✅

### 2281. Sum of Total Strength of Wizards https://leetcode.com/problems/sum-of-total-strength-of-wizards/ 

As the ruler of a kingdom, you have an army of wizards at your command.

You are given a 0-indexed integer array strength, where strength[i] denotes the strength of the ith wizard. For a contiguous group of wizards (i.e. the wizards' strengths form a subarray of strength), the total strength is defined as the product of the following two values:

The strength of the weakest wizard in the group.
The total of all the individual strengths of the wizards in the group.
Return the sum of the total strengths of all contiguous groups of wizards. Since the answer may be very large, return it modulo 109 + 7.

A subarray is a contiguous non-empty sequence of elements within an array.
```
Example 1:

Input: strength = [1,3,1,2]
Output: 44
Explanation: The following are all the contiguous groups of wizards:
- [1] from [1,3,1,2] has a total strength of min([1]) * sum([1]) = 1 * 1 = 1
- [3] from [1,3,1,2] has a total strength of min([3]) * sum([3]) = 3 * 3 = 9
- [1] from [1,3,1,2] has a total strength of min([1]) * sum([1]) = 1 * 1 = 1
- [2] from [1,3,1,2] has a total strength of min([2]) * sum([2]) = 2 * 2 = 4
- [1,3] from [1,3,1,2] has a total strength of min([1,3]) * sum([1,3]) = 1 * 4 = 4
- [3,1] from [1,3,1,2] has a total strength of min([3,1]) * sum([3,1]) = 1 * 4 = 4
- [1,2] from [1,3,1,2] has a total strength of min([1,2]) * sum([1,2]) = 1 * 3 = 3
- [1,3,1] from [1,3,1,2] has a total strength of min([1,3,1]) * sum([1,3,1]) = 1 * 5 = 5
- [3,1,2] from [1,3,1,2] has a total strength of min([3,1,2]) * sum([3,1,2]) = 1 * 6 = 6
- [1,3,1,2] from [1,3,1,2] has a total strength of min([1,3,1,2]) * sum([1,3,1,2]) = 1 * 7 = 7
The sum of all the total strengths is 1 + 9 + 1 + 4 + 4 + 4 + 3 + 5 + 6 + 7 = 44.
Example 2:

Input: strength = [5,4,6]
Output: 213
Explanation: The following are all the contiguous groups of wizards: 
- [5] from [5,4,6] has a total strength of min([5]) * sum([5]) = 5 * 5 = 25
- [4] from [5,4,6] has a total strength of min([4]) * sum([4]) = 4 * 4 = 16
- [6] from [5,4,6] has a total strength of min([6]) * sum([6]) = 6 * 6 = 36
- [5,4] from [5,4,6] has a total strength of min([5,4]) * sum([5,4]) = 4 * 9 = 36
- [4,6] from [5,4,6] has a total strength of min([4,6]) * sum([4,6]) = 4 * 10 = 40
- [5,4,6] from [5,4,6] has a total strength of min([5,4,6]) * sum([5,4,6]) = 4 * 15 = 60
The sum of all the total strengths is 25 + 16 + 36 + 36 + 40 + 60 = 213. 
``` 
> 这个solution 比较全面 
https://leetcode.com/problems/sum-of-total-strength-of-wizards/discuss/2061985/JavaC%2B%2BPython-One-Pass-Solution


```python
class Solution:
    def totalStrength(self, strength: List[int]) -> int:
        
        # brute force, time o(n^2), space o(1)， 超时
        total = 0
        for i in range(len(strength)):
            for j in range(i + 1, len(strength)+1):
                subarray = strength[i:j]
                # print(subarray)
                total += min(subarray) * sum(subarray)
        return total % (10**9 + 7)
            
    # stack 单调栈
    def totalStrength(self, A):
        res, ac, mod, stack, acc = 0, 0, 10 ** 9 + 7, [-1], [0]
        A += [0]
        for r, a in enumerate(A):
            ac += a
            acc.append(ac + acc[-1])
            while stack and A[stack[-1]] > a:
                i = stack.pop()
                l = stack[-1]
                lacc = acc[i] - acc[max(l, 0)]
                racc = acc[r] - acc[i]
                ln, rn = i - l, r - i
                res += A[i] * (racc * ln - lacc * rn) % mod
            stack.append(r)
        return res % mod
``` 


