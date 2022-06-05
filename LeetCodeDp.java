import org.junit.Test;

import java.util.List;

public class LeetCodeDp {

    // dp: dynamic programming 动态规划
    // no.78 subset
    //    输入：nums = [1,2,3]
    //    输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
    public List<List<Integer>> subsets(int[] nums) {

        int[] num = {1, 2, 3};
        return null;


    }

    // no.139 word break


//    91. 解码方法
//    一条包含字母 A-Z 的消息通过以下映射进行了 编码 ：
//
//            'A' -> "1"
//            'B' -> "2"
//            ...
//            'Z' -> "26"
//    要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，"11106" 可以映射为：
//
//            "AAJF" ，将消息分组为 (1 1 10 6)
//    "KJF" ，将消息分组为 (11 10 6)
//    注意，消息不能分组为  (1 11 06) ，因为 "06" 不能映射为 "F" ，这是由于 "6" 和 "06" 在映射中并不等价。
//
//    给你一个只含数字的 非空 字符串 s ，请计算并返回 解码 方法的 总数 。
//
//    题目数据保证答案肯定是一个 32 位 的整数。
//
//    输入：s = "12"
//    输出：2
//    解释：它可以解码为 "AB"（1 2）或者 "L"（12）。

    Integer[] memo;

    public int numDecodings(String s) {

        int n = s.length();
        memo = new Integer[n + 1];
        return dfs91(s, n);

    }

    public int dfs91(String s, int n) {

        if (n == 0)
            return 1;
        if (n == 1)
            return s.charAt(0) == '0' ? 0 : 1;
        if (memo[n] != null)
            return memo[n];

        int res = 0;
        char x = s.charAt(n - 1);
        char y = s.charAt(n - 2);

        if (x != '0')
            res += dfs91(s, n - 1);

        int xy = (y - '0') * 10 + (x - '0');
        if (xy >= 10 && xy <= 26) {
            res += dfs91(s, n - 2);
        }

        memo[n] = res;
        return res;

    }

//    96. 不同的二叉搜索树
//    给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？
//    返回满足题意的二叉搜索树的种数。
//
//    输入：n = 3
//    输出：5

    //    Integer[] memo;
    public int numTrees(int n) {

        memo = new Integer[n + 1];

        return dfs96(n);
    }

    public int dfs96(int n) {

        if (n <= 1)
            return 1;
        if (memo[n] != null)
            return memo[n];

        int res = 0;
        for (int i = 1; i <= n; i++) {
            int left = dfs96(i - 1);
            int right = dfs96(n - i);
            res += left * right;
        }
        memo[n] = res;

        return res;


    }


    @Test
    public void charTest() {
        String a = "123445";
        int m = a.charAt(0);
        System.out.println(m);
        System.out.println('9' - '0');
        System.out.println(m - '0' == 1);
        System.out.println(m * 10);

        String aa = "12";
        String bb = "12";
        System.out.println(aa + bb);
    }


//    63. 不同路径 II
//    一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
//
//    机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish”）。
//
//    现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
//
//    网格中的障碍物和空位置分别用 1 和 0 来表示。

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null)
            return 0;
        if (obstacleGrid[0][0] == 1)
            return 0;
        int i = obstacleGrid.length;
        int j = obstacleGrid[0].length;

        int[][] dp = new int[i + 1][j + 1];
        dp[0][0] = 1;
        for (int m = 1; m < i; m++) {
            if (obstacleGrid[m][0] == 1) {
                dp[m][0] = 0;
            } else {
                dp[m][0] = dp[m - 1][0];
            }
        }
        for (int m = 1; m < j; m++) {
            if (obstacleGrid[0][m] == 1) {
                dp[0][m] = 0;
            } else {
                dp[0][m] = dp[0][m - 1];
            }
        }
        for (int m = 1; m < i; m++) {
            for (int n = 1; n < j; n++) {

                if (obstacleGrid[m][n] == 1) {
                    dp[m][n] = 0;
                } else {
                    dp[m][n] = dp[m - 1][n] + dp[m][n - 1];
                }
            }
        }
        return dp[i][j];
    }


//    1143. 最长公共子序列
//    给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。
//
//    一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
//
//    例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
//    两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。
//
//
//
//    示例 1：
//
//    输入：text1 = "abcde", text2 = "ace"
//    输出：3
//    解释：最长公共子序列是 "ace" ，它的长度为 3 。

    Integer memo1[][];

    public int longestCommonSubsequence(String text1, String text2) {

        int i = text1.length();
        int j = text2.length();
        memo1 = new Integer[i][j];
        return dfs1134(text1, text2, i - 1, j - 1);

    }

    public int dfs1134(String text1, String text2, int i, int j) {

        if (i < 0 || j < 0)
            return 0;
        if (memo1[i][j] != null)
            return memo1[i][j];
        int num = 0;
        if (text1.charAt(i) == text2.charAt(j)) {
            num = dfs1134(text1, text2, i - 1, j - 1) + 1;
        } else {
            num = Math.max(dfs1134(text1, text2, i, j - 1), dfs1134(text1, text2, i - 1, j));
        }
        return memo1[i][j] = num;
    }

    //    877. 石子游戏
//    Alice 和 Bob 用几堆石子在做游戏。一共有偶数堆石子，排成一行；每堆都有 正 整数颗石子，数目为 piles[i] 。
//
//    游戏以谁手中的石子最多来决出胜负。石子的 总数 是 奇数 ，所以没有平局。
//
//    Alice 和 Bob 轮流进行，Alice 先开始 。 每回合，玩家从行的 开始 或 结束 处取走整堆石头。 这种情况一直持续到没有更多的石子堆为止，此时手中 石子最多 的玩家 获胜 。
//
//    假设 Alice 和 Bob 都发挥出最佳水平，当 Alice 赢得比赛时返回 true ，当 Bob 赢得比赛时返回 false 。
//
//
//    示例 1：
//    输入：piles = [5,3,4,5]
//    输出：true
    Integer[][] memo2;

    public boolean stoneGame(int[] piles) {
        int n = piles.length;
        memo2 = new Integer[n][n];

        return dfs877(piles, 0, n - 1) > 0;
    }

    public int dfs877(int[] piles, int i, int j) {

        if (i > j)
            return 0;
        if (i == j)
            return piles[i];
        if (memo2[i][j] != null)
            return memo2[i][j];

        int res = Math.max(piles[i] - dfs877(piles, i + 1, j),
                piles[j] - dfs877(piles, i, j - 1));

        return res;
    }

    // 正向的dp
    public boolean stoneGame1(int[] piles) {
        int n = piles.length;
        int[][] memo = new int[n][n];

        for (int i = 0; i < n; i++) {
            memo[i][i] = piles[i];
        }

        for (int l = 2; l <= n; l++) {
            for (int i = 0; i <= n - l; i++) {
                int j = i + l - 1;
                memo[i][j] = Math.max(piles[i] - memo[i + 1][j],
                        piles[j] - memo[i][j - 1]);
            }
        }
        return memo[0][n - 1] > 0;
    }


    /*#
    # 5. 最长回文子串
    # 给你一个字符串 s，找到 s 中最长的回文子串。
    #
    # 示例 1：
    # 输入：s = "babad"
    # 输出："bab"
    # 解释："aba" 同样是符合题意的答案。*/

    Boolean memo5[][];
    String lps = "";

    public String longestPalindrome(String s) {
        int i = 0, n = s.length();

        memo5 = new Boolean[n][n];

        dfs2(s, 0, n-1);

        return lps;
    }


    public Boolean dfs2(String s, int i, int j) {

        if (i >= j) {
            memo5[i][j] = true;
            if (lps.length() < j - i + 1) {
                lps = s.substring(i, j + 1);
            }
            return true;
        }
        if (memo5[i][j] != null)
            return memo5[i][j];

        Boolean res = false;
        if (s.charAt(i) == s.charAt(j) && dfs2(s, i + 1, j - 1)) {
            res = true;
            if (lps.length() < j - i + 1)
                lps = s.substring(i, j + 1);
        } else {
            dfs2(s, i + 1, j);
            dfs2(s, i, j - 1);
        }
        return memo5[i][j] = res;


    }


//    188. 买卖股票的最佳时机 IV
//    给定一个整数数组 prices ，它的第 i 个元素 prices[i] 是一支给定的股票在第 i 天的价格。
//
//    设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。
//
//    注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
//
//
//
//    示例 1：
//
//    输入：k = 2, prices = [2,4,1]
//    输出：2
//    解释：在第 1 天 (股票价格 = 2) 的时候买入，在第 2 天 (股票价格 = 4) 的时候卖出，这笔交易所能获得利润 = 4-2 = 2 。

//    int[][][] memo3;
//
//    public int maxProfit(int k, int[] prices) {
////        memo3[i][k][0]
//        int n = prices.length;
//        memo3 = new int[n + 1][k + 1][2];
//
//        dff188(memo3, k, prices);
//
//        return memo3[n][k][0];
//    }
//
//    public void dff188(int[][][] memo3, int k, int[] price) {
//        memo3[0][0][0] = 0;
//        memo3[0][0][1] = -price[0];
//        memo3[0][1][0] =0;
//
//        for (int i = 1; i < price.length; i++) {
//            for (int j = 1; j < k; j++) {
//                memo3[i][j][0] = Math.max(memo3[i - 1][j][0], memo3[i - 1][j][1] + price[i]);
//                memo3[i][j][1] = Math.max(memo3[i - 1][j][1], memo3[i - 1][j - 1][0] - price[i]);
//            }
//        }
//    }
//
//    @Test
//    public void testDo() {
//        int k = 2;
//        int[] prices = new int[]{2, 4, 1};
//
//        int m = maxProfit(k, prices);
//        System.out.println(m);
//    }


}
