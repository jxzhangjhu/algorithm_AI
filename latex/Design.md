# Design 


## 入门design

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






