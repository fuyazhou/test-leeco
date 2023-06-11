# # [LeetCode] 72. Edit Distance 编辑距离
# #
# #
# # Given two words word1 and word2, find the minimum number of operations required to convert word1 to word2.
# #
# # You have the following 3 operations permitted on a word:
# #
# # Insert a character
# # Delete a character
# # Replace a character
#
# # class Solution(object):
# #     def minDistance(self, word1, word2):
# #         """
# #         :type word1: str
# #         :type word2: str
# #         :rtype: int
# #         """
# #         if not word1 or not word2: return max(len(word1), len(word2))
# #
# #         m, n = len(word1) + 1, len(word2) + 1
# #         dp = [[0 for _ in range(n)] for _ in range(m)]
# #         for i in range(m):
# #             dp[i][0] = i
# #
# #         for j in range(n):
# #             dp[0][j] = j
# #
# #         for i in range(1, m):
# #             for j in range(1, n):
# #                 dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
# #                 if word1[i - 1] == word2[j - 1]:
# #                     dp[i][j] = min(dp[i][j], dp[i - 1][j - 1])
# #         return dp[m-1][n-1]
#
# def minDistance(word1, word2):
#     m = len(word1) + 1
#     n = len(word2) + 1
#
#     dp = [[0 for _ in range(n)] for _ in range(m)]
#
#     for i in range(m):
#         dp[i][0] = i
#     for i in range(n):
#         dp[0][i] = i
#
#     for i in range(1, m):
#         for j in range(1, n):
#             if word1[i - 1] == word2[j - 1]:
#                 dp[i][j] = dp[i - 1][j - 1]
#             else:
#                 dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
#
#     print(dp[m - 1][n - 1])
#
#     print(dp)
#
#
# # minDistance("horse", "ros")
#
#
# # 78. 子集
# # 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
# #
# # 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集
# #
# # 示例 1：
# #
# # 输入：nums = [1,2,3]
# # 输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
#
# # def subsets(self, nums: List[int]) -> List[List[int]]:
# def subsets(nums):
#     if len(nums) == 0:
#         return 0
#     # nums = [1, 2, 3]
#
#     state = []
#     res = []
#     isvited = set();
#
#     def back(nums, state):
#         # if len(nums) == 0:
#         #     return
#         # if set(state) in isvited:
#         #     return
#         # isvited.add(set(state))
#         res.append(state)
#         for i in range(0, len(nums)):
#             back(nums[i + 1:], state + [nums[i]])
#
#     back(nums, state)
#
#     print(res)
#     return res
#
#
# # subsets([1, 2, 3])
#
#
# # 139. 单词拆分
# # 给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。
# #
# # 注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。
# #
# #
# #
# # 示例 1：
# #
# # 输入: s = "leetcode", wordDict = ["leet", "code"]
# # 输出: true
# # 解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
#
# # s = "leetcode", wordDict = ["leet", "code"]
# def wordBreak(s, wordDict):
#     if len(s) == 0:
#         return False
#
#     res = [False]
#     state = ""
#
#     def back(state):
#         if len(state) == 0:
#             res[0] = True
#             return
#         for i in range(1, len(state)):
#             if s[0:i] in wordDict:
#                 state = state[i:]
#                 back(state)
#
#     back(state)
#     print(res)
#     return res
#
#
# # wordBreak("leetcode", ["leet", "code"])
#
#
# #
# # 5. 最长回文子串
# # 给你一个字符串 s，找到 s 中最长的回文子串。
# #
# # 示例 1：
# # 输入：s = "babad"
# # 输出："bab"
# # 解释："aba" 同样是符合题意的答案。
# def longestPalindrome(self, s: str) -> str:
#     def isPalindrome(s):
#         if s == s[::-1]:
#             return True
#         return False
#
#     res = [""]
#     state = ""
#
#     def back(s):
#         if len(s) == 0:
#             return
#         for i in range(1, len(s) + 1):
#             if isPalindrome(s[0:i]):
#                 if len(s[0:i]) > len(res[0]):
#                     res[0] = s[0:i]
#             back(s[i:])
#         return res
#
#     res = back(s)
#     return res[0]
#
#
# def longestPalindrome(self, s: str) -> str:
#     res = [""]
#
#     def isPalindrome(mid):
#         for i in range(0, len(s) - mid):
#             tem = s[i:mid]
#             if tem == tem[::-1]:
#                 res[0] = tem
#                 return True
#             return False
#
#     i = 0
#     j = len(s)
#
#     while (i < j):
#         mid = (i + j) / 2
#         if isPalindrome(mid):
#             i = mid
#         else:
#             j = mid
#
#     return res[0]

#
# 15. 三数之和
# 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
#
# 注意：答案中不可以包含重复的三元组。
#
#
#
# 示例 1：
#
# 输入：nums = [-1,0,1,2,-1,-4]
# 输出：[[-1,-1,2],[-1,0,1]]
import collections
import functools

import numpy as np


def threeSum1(nums):
    if len(nums) == 0:
        return []
    res = []
    state = []

    s = set();
    nums.sort()

    def back(nums, state):

        if len(state) == 3 and sum(state) == 0:
            state.sort()
            if tuple(state) not in s:
                s.add(tuple(state))
                res.append(state)
                return
        if len(nums) < 1 or len(state) > 3:
            return

        for i in range(0, len(nums)):
            # if nums[i] <= 0:
            back(nums[i + 1:], state + [nums[i]])

        return res

    return back(nums, state)


def threeSum(nums):
    n = len(nums)
    res = []
    if (not nums or n < 3):
        return []
    nums.sort()
    res = []
    for i in range(n):
        if (nums[i] > 0):
            return res
        if (i > 0 and nums[i] == nums[i - 1]):
            continue
        L = i + 1
        R = n - 1
        while (L < R):
            if (nums[i] + nums[L] + nums[R] == 0):
                res.append([nums[i], nums[L], nums[R]])
                while (L < R and nums[L] == nums[L + 1]):
                    L = L + 1
                while (L < R and nums[R] == nums[R - 1]):
                    R = R - 1
                L = L + 1
                R = R - 1
            elif (nums[i] + nums[L] + nums[R] > 0):
                R = R - 1
            else:
                L = L + 1
    return res


# 46. 全排列
# 输入：nums = [1,2,3]
# 输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

def permute(nums):
    res = []
    state = []

    def back(state, nums):
        if len(nums) == 0:
            res.append(state)
        for i in range(len(nums)):
            back(state + [nums[i]], nums[:i] + nums[i + 1:])

    back(state, nums)
    return res;


num1 = "234"
num2 = "67237"


def addStrings(num1, num2):
    res = ""
    l1 = len(num1) - 1
    l2 = len(num2) - 1
    carry = 0

    while l1 > 0 or l2 > 0:
        a1 = int(num1[l1]) if l1 > 0 else 0;
        a2 = int(num2[l2]) if l2 > 0 else 0;
        l1 = l1 - 1
        l2 = l2 - 1
        sub = (a1 + a2 + carry) % 10
        carry = (a1 + a2 + carry) // 10

        res = str(sub) + res;

    if carry > 0:
        res = str(carry) + res
    return res


# "hello world"
def reverseWords(s):
    while "  " in s:
        s = s.replace("  ", " ")
    s = s[::-1]
    s = s.split(" ")
    temp = ""
    for i in range(len(s)):
        temp = temp + s[i][::-1] + " "
    temp = temp.strip()
    print(temp)
    return temp


# reverseWords(" hello   world ")


# 239. 滑动窗口最大值
def maxSlidingWindow(nums, k):
    length = len(nums)
    res = []

    if (length <= k):
        tem = max(nums)
        res.append(tem)
        return res

    for i in range(0, length - k + 1):
        tem = max(nums[i:i + k])
        res.append(tem)
    return res


# maxSlidingWindow([1], 1)


# 56. 合并区间
def merge(nums):
    nums.sort(key=lambda x: x[0])
    merged = []

    for num in nums:
        if not merged or merged[-1][1] < num[0]:
            merged.append(num)
        else:
            merged[-1][1] = max(merged[-1][1], num[1])

    print(merged)


# intervals = [[1, 3], [0, 6], [8, 10], [15, 18]]
# merge(intervals)


# 322. 零钱兑换
def coinChange(coins, amount):
    if len(coins) < 1:
        return -1;

    state = []
    allsum = 0
    res = [-1]
    m = set()

    def back(coins, state):
        if sum(state) == amount:
            if res[0] == -1:
                res[0] = len(state)
            else:
                res[0] = min(res[0], len(state))
        if sum(state) > amount:
            return

        if res[0] != -1 and len(state) > res[0]:
            return

        if tuple(state) in m:
            return
        else:
            m.add(tuple(state))

        for i in range(0, len(coins)):
            back(coins, state + [coins[i]])

    back(coins, state)
    return res[0]


def coinChange1(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1


# coins = [3, 1, 2, 5]
# coins = sorted(coins, reverse=True)
# amount = 110
# res = coinChange1(coins, amount)
# print(res)


def subsets(nums):
    res = []
    if not nums or len(nums) == 0:
        return res;

    state = []
    memo = set()

    def back(state, nums):

        # memo.add(tuple(state))
        res.append(state)

        if len(nums) == 0:
            return

        for i in range(0, len(nums)):
            back(state + [nums[i]], nums[i:])

    back(state, nums)
    return res


def subsetsWithDup(nums):
    res = []  # 定义全局变量保存最终结果
    state = []  # 定义状态变量保存当前状态
    s = set()  # 定义条件变量（一般条件变量就是题目直接给的参数）

    def back(state, q):
        if tuple(state) in s:  # 不满足合法条件（可以说是剪枝）
            return
        else:  # 状态满足最终要求
            s.add(tuple(state))
            res.append([i for i in state])  # 加入结果
        # 主要递归过程，一般是带有 循环体 或者 条件体
        for i in range(q, len(nums)):
            back(state + [nums[i]], i + 1)

    nums.sort()
    back(state, 0)
    return res


# nums = [1, 2, 3]
# m = subsetsWithDup(nums)
# print(m)
def longestCommonPrefix(strs):
    if strs is None:
        return ""
    strsLen = [len(i) for i in strs]
    minLen = min(strsLen)
    res = ""
    flag = False
    for i in range(0, minLen):
        cur = strs[0][0:i + 1]
        for str1 in strs:
            if str1[0:i + 1] != cur:
                flag = True
                break
        if not flag:
            res = cur
        if flag:
            break
    return res


# longestCommonPrefix(["ab", "a"])


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def isPalindrome(head):
    res = []
    if head is not None:
        res.append(head.val)

    if res == res[::-1]:
        return True
    else:
        return False


# 179. 最大数

def largestNumber(nums):
    strs = map(str, nums)

    def cmp(a, b):
        if a + b == b + a:
            return 0
        elif a + b > b + a:
            return 1
        else:
            return -1

    strs = sorted(strs, key=functools.cmp_to_key(cmp), reverse=True)
    return ''.join(strs) if strs[0] != '0' else '0'


def removeKdigits(nums, k):
    if nums is None or len(nums) <= k:
        return "0"

    nums = sorted(nums)
    nums = nums[0:len(nums) - k]

    res = "".join(nums)

    return str(int(res))


# 316 去除重复字符
# 输入：s = "bcabc"
# 输出："abc"
def removeDuplicateLetters(s: str) -> str:
    stack = []
    remain_counter = collections.Counter(s)

    for c in s:
        if c not in stack:
            while stack and stack[-1] > c and remain_counter[stack[-1]] > 0:
                stack.pop()
            stack.append(c)
        remain_counter[c] -= 1
    return "".join(stack)


#
# s = "cbacdcbc"
# res = removeDuplicateLetters(s)
# print(res)

def maxNumber(nums1, nums2, k):
    def pick_max(nums, k):
        stack = []
        drop = len(nums) - k
        for num in nums:
            while drop and stack and stack[-1] < num:
                stack.pop()
                drop -= 1
            stack.append(num)
        return stack[:k]

    def merge(A, B):
        ans = []
        while A or B:
            bigger = A if A > B else B
            ans.append(bigger[0])
            bigger.pop(0)
        return ans

    return max(merge(pick_max(nums1, i), pick_max(nums2, k - i)) for i in range(k + 1) if
               i <= len(nums1) and k - i <= len(nums2))


def subarraySum(nums, k):
    res = [0]
    if nums is None or len(nums) == 0:
        return res[0]

    nums.sort()

    i = 0
    j = 0

    while j < len(nums):

        while sum(nums[i:j + 1]) > k and i <= j:
            i += 1

        if sum(nums[i:j + 1]) == k:
            res[0] += 1

        j = j + 1

    return res[0]


#
# nums = [1, 1, 1]
# k = 2
# res = subarraySum(nums, k)
# print(res)


def wordBreak(s, wordDict):
    if s in wordDict:
        return True
    if len(wordDict) < 1:
        return False
    if s is None:
        return False

    res = [False]

    def back(s, wordDict):
        if res[0]:
            return

        if len(s) == 0:
            res[0] = True
            return

        for i in range(0, len(s)):
            if s[0:i + 1] in wordDict:
                back(s[i + 1:], wordDict)

    back(s, wordDict)
    return res[0]


#
# s = "leetcode"
# wordDict = ["leet", "code"]
# res=wordBreak(s,wordDict)
# print(res)


# 47 全排列
def permuteUnique(nums):
    res = []
    if nums is None:
        return res

    state = []

    set1 = set()

    def back(state, nums):
        if tuple(state) in set1:
            return
        set1.add(tuple(state))

        if len(nums) == 0:
            res.append(state)
            return

        for i in range(len(nums)):
            back(state + [nums[i]], nums[:i] + nums[i + 1:])

    back(state, nums)
    return res


# nums = [1, 1, 2]
# res = permuteUnique(nums)
# print(res)


def myPow(x, n):
    if n == 0:
        return 1
    if x == 0:
        return 0

    if_negitve = False
    if n < 0:
        if_negitve = True
        n = 0 - n

    res = 1
    x_contribute = x
    while n > 0:
        if n % 2 == 1:
            res *= x_contribute

        x_contribute = x_contribute * x_contribute
        n = n // 2

    if if_negitve:
        res = 1 / res

    return res


def reversePairs(nums):
    if nums is None or len(nums) <= 1:
        return 0

    res = 0
    for i in range(len(nums)):
        for j in range(i + 1, nums):
            if nums[i] > nums[j]:
                res += 1
    return res


def reverse(x):
    if x is None:
        return x
    neg = False
    if x < 0:
        neg = True
        x = -x
    s = list(str(x))
    i = 0
    j = len(s)-1
    while i < j:
        temp = s[i]
        s[i] = s[j]
        s[j] = temp
        i += 1
        j -= 1

    if neg:
        x = -int("".join(s))
        return 0 if x < -2 ** 31 else x
    else:
        x = int("".join(s))
        return 0 if x > 2 ** 31 - 1 else x
