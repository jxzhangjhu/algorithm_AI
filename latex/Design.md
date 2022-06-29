# Design 


## 入门design

### 232. Implement Queue using Stacks https://leetcode.com/problems/implement-queue-using-stacks/ 

Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (push, peek, pop, and empty).

Implement the MyQueue class:

void push(int x) Pushes element x to the back of the queue.
int pop() Removes the element from the front of the queue and returns it.
int peek() Returns the element at the front of the queue.
boolean empty() Returns true if the queue is empty, false otherwise.
Notes:

You must use only standard operations of a stack, which means only push to top, peek/pop from top, size, and is empty operations are valid.
Depending on your language, the stack may not be supported natively. You may simulate a stack using a list or deque (double-ended queue) as long as you use only a stack's standard operations.
 
```
Example 1:

Input
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]
Output
[null, null, null, 1, 1, false]

Explanation
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); // return false
```
> 这个题挺好的，多练练，这样对stack 和queue的一些操作都有更深刻的理解！
```python
# # Approach 1: push O(n), pop O(1)
# class MyQueue:
#     def __init__(self):
#         self.s1 = []
#         self.s2 = []

#     def push(self, x):
#         while self.s1:
#             self.s2.append(self.s1.pop())
#         self.s1.append(x)
#         while self.s2:
#             self.s1.append(self.s2.pop())

#     def pop(self):
#         return self.s1.pop()

#     def peek(self):
#         return self.s1[-1]

#     def empty(self):
#         return not self.s1
    
# Approach 2: push O(1), pop amortized O(1)
class MyQueue:
    def __init__(self):
        self.s1 = []
        self.s2 = []

    def push(self, x):
        self.s1.append(x)

    def pop(self):
        self.peek()
        return self.s2.pop()

    def peek(self):
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
        return self.s2[-1]        

    def empty(self):
        return not self.s1 and not self.s2    

```



### 155. Min Stack 

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:

MinStack() initializes the stack object.
void push(int val) pushes the element val onto the stack.
void pop() removes the element on the top of the stack.
int top() gets the top element of the stack.
int getMin() retrieves the minimum element in the stack.
You must implement a solution with O(1) time complexity for each function.

```
Example 1:

Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output
[null,null,null,null,-3,null,0,-2]

Explanation
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2
```
> 很好的design题，和上面queue using stack 和这个题，更熟悉如何操作queue and stack，后面还有很多design用heap的
```python
class MinStack(object):
    # 需要理解题目的含义，就是要设计一个stack，能够直接返回最小值，这个不清楚之前！ 
    # 如果不理解就要问
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        if not self.stack:
            self.stack.append((x, x))
        else:
            self.stack.append((x, min(x, self.stack[-1][1])))
    def pop(self):
        """
        :rtype: void
        """
        self.stack.pop()
    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1][0]
    def getMin(self):
        """
        :rtype: int
        """
        return self.stack[-1][1]
# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```


### 716. Max Stack 
Design a max stack data structure that supports the stack operations and supports finding the stack's maximum element.

Implement the MaxStack class:

MaxStack() Initializes the stack object.
void push(int x) Pushes element x onto the stack.
int pop() Removes the element on top of the stack and returns it.
int top() Gets the element on the top of the stack without removing it.
int peekMax() Retrieves the maximum element in the stack without removing it.
int popMax() Retrieves the maximum element in the stack and removes it. If there is more than one maximum element, only remove the top-most one.
You must come up with a solution that supports O(1) for each top call and O(logn) for each other call? 

```
Example 1:

Input
["MaxStack", "push", "push", "push", "top", "popMax", "top", "peekMax", "pop", "top"]
[[], [5], [1], [5], [], [], [], [], [], []]
Output
[null, null, null, null, 5, 5, 1, 5, 1, 5]

Explanation
MaxStack stk = new MaxStack();
stk.push(5);   // [5] the top of the stack and the maximum number is 5.
stk.push(1);   // [5, 1] the top of the stack is 1, but the maximum is 5.
stk.push(5);   // [5, 1, 5] the top of the stack is 5, which is also the maximum, because it is the top most one.
stk.top();     // return 5, [5, 1, 5] the stack did not change.
stk.popMax();  // return 5, [5, 1] the stack is changed now, and the top is different from the max.
stk.top();     // return 1, [5, 1] the stack did not change.
stk.peekMax(); // return 5, [5, 1] the stack did not change.
stk.pop();     // return 1, [5] the top of the stack and the max element is now 5.
stk.top();     // return 5, [5] the stack did not change.
```
> hard题，和上面的对比来搞，min stack and max stack？
```python
class MaxStack:

    def __init__(self):
        self.stack = []

    def push(self, x: int) -> None:
        self.stack.append(x)

    def pop(self) -> int:
        return self.stack.pop()
        
    def top(self) -> int:
        return self.stack[-1]

    def peekMax(self) -> int:
        return max(self.stack)

    def popMax(self) -> int:
        max_element = max(self.stack)
        
        index = len(self.stack) - 1
        while self.stack[index] != max_element:
            index -= 1
        del self.stack[index]
        
        return max_element
```





### 1603. Design Parking System https://leetcode.com/problems/design-parking-system/ 

Design a parking system for a parking lot. The parking lot has three kinds of parking spaces: big, medium, and small, with a fixed number of slots for each size.

Implement the ParkingSystem class:

ParkingSystem(int big, int medium, int small) Initializes object of the ParkingSystem class. The number of slots for each parking space are given as part of the constructor.
bool addCar(int carType) Checks whether there is a parking space of carType for the car that wants to get into the parking lot. carType can be of three kinds: big, medium, or small, which are represented by 1, 2, and 3 respectively. A car can only park in a parking space of its carType. If there is no space available, return false, else park the car in that size space and return true.
 
```
Example 1:

Input
["ParkingSystem", "addCar", "addCar", "addCar", "addCar"]
[[1, 1, 0], [1], [2], [3], [1]]
Output
[null, true, true, false, false]

Explanation
ParkingSystem parkingSystem = new ParkingSystem(1, 1, 0);
parkingSystem.addCar(1); // return true because there is 1 available slot for a big car
parkingSystem.addCar(2); // return true because there is 1 available slot for a medium car
parkingSystem.addCar(3); // return false because there is no available slot for a small car
parkingSystem.addCar(1); // return false because there is no available slot for a big car. It is already occupied.
```

```python
# class ParkingSystem:

#     def __init__(self, big: int, medium: int, small: int):
#         self.big = big 
#         self.medium = medium
#         self.small = small

#     def addCar(self, carType: int) -> bool:
        
#         if carType == 1:
#             if self.big:
#                 self.big -= 1
#                 return True
#             else:
#                 return False 
#         if carType == 2:
#             if self.medium:
#                 self.medium -= 1
#                 return True
#             else:
#                 return False 
#         if carType == 3:
#             if self.small:
#                 self.small -= 1
#                 return True
#             else:
#                 return False         

class ParkingSystem:

    def __init__(self, big: int, medium: int, small: int):
        self.num_dic = {1:big,2:medium,3:small}

    def addCar(self, carType: int) -> bool:
        if self.num_dic[carType] > 0:
            self.num_dic[carType] -=1
            return True
        return False


# Your ParkingSystem object will be instantiated and called as such:
# obj = ParkingSystem(big, medium, small)
# param_1 = obj.addCar(carType)
# Your ParkingSystem object will be instantiated and called as such:

obj = ParkingSystem(1,1,0)
param_1 = obj.addCar(1)
print(param_1)
param_2 = obj.addCar(2)
print(param_2)
param_3 = obj.addCar(3)
print(param_3)
param_4 = obj.addCar(1)

True
True
False
False

```




## Hashtable 

### 359. Logger Rate Limiter  
Design a logger system that receives a stream of messages along with their timestamps. Each unique message should only be printed at most every 10 seconds (i.e. a message printed at timestamp t will prevent other identical messages from being printed until timestamp t + 10).

All messages will come in chronological order. Several messages may arrive at the same timestamp.

Implement the Logger class:

Logger() Initializes the logger object.
bool shouldPrintMessage(int timestamp, string message) Returns true if the message should be printed in the given timestamp, otherwise returns false.
```
Input
["Logger", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage"]
[[], [1, "foo"], [2, "bar"], [3, "foo"], [8, "bar"], [10, "foo"], [11, "foo"]]
Output
[null, true, true, false, false, false, true]

Explanation
Logger logger = new Logger();
logger.shouldPrintMessage(1, "foo");  // return true, next allowed timestamp for "foo" is 1 + 10 = 11
logger.shouldPrintMessage(2, "bar");  // return true, next allowed timestamp for "bar" is 2 + 10 = 12
logger.shouldPrintMessage(3, "foo");  // 3 < 11, return false
logger.shouldPrintMessage(8, "bar");  // 8 < 12, return false
logger.shouldPrintMessage(10, "foo"); // 10 < 11, return false
logger.shouldPrintMessage(11, "foo"); // 11 >= 11, return true, next allowed timestamp for "foo" is 11 + 10 = 21
```
> 主要是运用hashtable，但是别忘了更新，要读懂题

```python
# time o(1), space o(M), M is the size of all incoming message. 
class Logger:
    def __init__(self):
        self.hashmap = {}
    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        if message in self.hashmap:
            if timestamp - self.hashmap[message] >= 10:
                self.hashmap[message] = timestamp # 这一步update 是必要的！
                return True
            else:
                return False
        else: 
            self.hashmap[message] = timestamp
            return True

# way 2 - queue + set 没想到，但是这个都是o(N) 不见得更优， official solution 
```

### 1146. Snapshot Array https://leetcode.com/problems/snapshot-array/ 
Implement a SnapshotArray that supports the following interface:

SnapshotArray(int length) initializes an array-like data structure with the given length.  Initially, each element equals 0.
void set(index, val) sets the element at the given index to be equal to val.
int snap() takes a snapshot of the array and returns the snap_id: the total number of times we called snap() minus 1.
int get(index, snap_id) returns the value at the given index, at the time we took the snapshot with the given snap_id
 
```
Example 1:

Input: ["SnapshotArray","set","snap","set","get"]
[[3],[0,5],[],[0,6],[0,0]]
Output: [null,null,0,null,5]
Explanation: 
SnapshotArray snapshotArr = new SnapshotArray(3); // set the length to be 3
snapshotArr.set(0,5);  // Set array[0] = 5
snapshotArr.snap();  // Take a snapshot, return snap_id = 0
snapshotArr.set(0,6);
snapshotArr.get(0,0);  // Get the value of array[0] with snap_id = 0, return 5
```
> 这种二分需要直接call package还是要自己写？ 
```python
class SnapshotArray:

    def __init__(self, length: int):
        # 初始化字典数组和 id
        self.arr = [{0: 0} for _ in range(length)]
        self.sid = 0

    def set(self, index: int, val: int) -> None:
        # 设置当前快照的元素值
        self.arr[index][self.sid] = val

    def snap(self) -> int:
        # 每次快照 id 加 1
        self.sid += 1
        # 返回上一个快照 id
        return self.sid - 1

    def get(self, index: int, snap_id: int) -> int:
        # 选择要查找的元素的字典
        d = self.arr[index]
        # 如果快照恰好存在, 直接返回
        if snap_id in d:
            return d[snap_id]
        # 不存在则进行二分搜索, 查找快照前最后一次修改
        k = list(d.keys())
        i = bisect.bisect_left(k, snap_id)
        return d[k[i - 1]]

# 作者：smoon1989
# 链接：https://leetcode.cn/problems/snapshot-array/solution/ha-xi-zi-dian-er-fen-cha-zhao-python3-by-smoon1989/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。       


# Your SnapshotArray object will be instantiated and called as such:
# obj = SnapshotArray(length)
# obj.set(index,val)
# param_2 = obj.snap()
# param_3 = obj.get(index,snap_id)


# class SnapshotArray(object):

#     def __init__(self, n):
#         self.A = [[[-1, 0]] for _ in range(n)]
#         self.snap_id = 0

#     def set(self, index, val):
#         self.A[index].append([self.snap_id, val])

#     def snap(self):
#         self.snap_id += 1
#         return self.snap_id - 1

#     def get(self, index, snap_id):
#         i = bisect.bisect(self.A[index], [snap_id + 1]) - 1
#         return self.A[index][i][1]
``` 






