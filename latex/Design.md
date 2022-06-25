# Design 



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




