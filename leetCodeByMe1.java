import org.junit.Test;

import java.util.*;

public class leetCodeByMe1 {

    //    153. 寻找旋转排序数组中的最小值
    public int findMin(int[] nums) {
        if (nums == null)
            return -1;

        int i = 0, j = nums.length - 1;
        int res = -1;

        while (i < j) {
            int mid = i + (j - i) / 2;
            if (nums[mid] < nums[j]) {
                j = mid;
            } else {
                i = mid + 1;
            }

        }
        return nums[i];
    }

    @Test
    public void ptest1() {
        int nums[] = new int[]{4, 5, 6, 7, 0, 1, 2};
        int min = findMin(nums);
        System.out.println(nums);
    }

    //    695. 岛屿的最大面积
    int[][] directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    public int maxAreaOfIsland(int[][] grid) {
        int res = 0;
        if (grid == null)
            return res;
        int rows = grid.length;
        int columns = grid[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (grid[i][j] == 1) {
                    res = Math.max(dfs695(grid, i, j), res);
                }
            }
        }
        return res;
    }

    public int dfs695(int[][] grid, int i, int j) {
        int temp = 1;
        grid[i][j] = 0;
        for (int[] dir : directions) {
            int m = i + dir[0];
            int n = j + dir[1];
            if (m >= 0 && m < grid.length && n >= 0 && n < grid[0].length && grid[m][n] == 1) {
                temp += dfs695(grid, m, n);
            }
        }
        return temp;
    }


    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }


    //    662. 二叉树最大宽度
    public int widthOfBinaryTree(TreeNode root) {
        int res = 0;
        if (root == null)
            return res;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int size = queue.size();
            res = Math.max(size, res);
            for (int i = 0; i < size; i++) {
                TreeNode temp = queue.poll();

                queue.offer(temp.left);
                queue.offer(temp.left);

            }
        }


        return res;
    }


    public class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }


    //    24. 两两交换链表中的节点
    public ListNode swapPairs1(ListNode head) {

        if (head == null || head.next == null)
            return head;

        ListNode newHead = head.next;
        head.next = swapPairs(newHead.next);
        newHead.next = head;
        return newHead;
    }

    public ListNode swapPairs(ListNode head) {
        ListNode dummyHead = new ListNode(0, head);
        ListNode temp = dummyHead;
        while (temp.next != null && temp.next.next != null) {
            ListNode node1 = temp.next;
            ListNode node2 = temp.next.next;
            temp.next = node2;
            node1.next = node2.next;
            node2.next = node1;
            temp = node1;
        }
        return dummyHead.next;
    }


    // 122 买卖股票的最佳时间
    public int maxProfit1(int[] prices) {
        int len = prices.length;
        int res = 0;

        if (len < 2)
            return res;

        for (int i = 1; i < len; i++) {
            if (prices[i] - prices[i - 1] > 0) {
                res += prices[i] - prices[i - 1];
            }
        }

        return res;
    }

    //使用动态规划的方法
    public int maxProfit(int[] prices) {

        int len = prices.length;
        int[][] dp = new int[len][2];  // 表示第 i 天交易完后手里有没有股票的最大利润

        dp[0][0] = 0;
        dp[0][1] = -prices[0];

        for (int i = 1; i < len; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][0] - prices[i], dp[i - 1][1]);
        }

        return dp[len - 1][0];

    }

    @Test
    public void test122() {
        int nums[] = new int[]{4, 5, 6, 7, 0, 1, 2};
        int min = maxProfit(nums);
        System.out.println(min);
    }


    //    198. 打家劫舍
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (nums.length == 1)
            return nums[0];
        if (nums.length == 2)
            return Math.max(nums[0], nums[1]);

        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);

        for (int i = 2; i < nums.length; i++) {

            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }

        return dp[nums.length - 1];

    }

    public int maxProduct(int[] nums) {
        if (nums == null || nums.length < -1)
            return 0;
        if (nums.length == 1)
            return nums[0];

        long[] pDp = new long[nums.length];
        long[] nDp = new long[nums.length];

        pDp[0] = nums[0];
        nDp[0] = nums[0];

        long max = nums[0];

        for (int i = 1; i < nums.length; i++) {
            pDp[i] = Math.max(Math.max(pDp[i - 1] * nums[i], nDp[i - 1] * nums[i]), nums[i]);
            nDp[i] = Math.min(Math.min(pDp[i - 1] * nums[i], nDp[i - 1] * nums[i]), nums[i]);

        }
        max = pDp[0];
        for (int i = 1; i < pDp.length; i++) {
            max = Math.max(max, pDp[i]);
        }
        return (int) max;
    }

    //    136. 只出现一次的数字
    public int singleNumber(int[] nums) {
        if (nums == null || nums.length < 1)
            return 0;
        if (nums.length == 1)
            return nums[0];

        int res = nums[0];
        for (int i = 1; i < nums.length; i++) {
            res = res ^ nums[i];
        }

        return res;


    }


    @Test
    public void test136() {
        int a = 2;
        int b = 1;
        int min = a & b;

        System.out.println(min);

    }

    class Node {
        int val;
        Node next;
        Node random;

        public Node(int val) {
            this.val = val;
            this.next = null;
            this.random = null;
        }
    }

    //    138. 复制带随机指针的链表
    public Node copyRandomList(Node head) {

        if (head == null)
            return null;

        Map<Node, Node> map = new HashMap<>();

        if (!map.containsKey(head)) {

            Node newHead = new Node(head.val);
            map.put(head, newHead);
            newHead.next = copyRandomList(head.next);
            newHead.random = copyRandomList(head.random);
        }
        return map.get(head);
    }

    //    209. 长度最小的子数组
    //    输入：target = 7, nums = [2,3,1,2,4,3]
    //    输出：2
    //    解释：子数组 [4,3] 是该条件下的长度最小的子数组。
    public int minSubArrayLen(int target, int[] nums) {
        int res = 0;
        if (nums == null || nums.length < 1)
            return res;
        int i = 0, j = 0;
        int sum = Integer.MAX_VALUE;
        while (j < nums.length) {
            sum += nums[j];

            while (sum >= target && i <= j) {
                res = Math.min(res, j - i + 1);
                sum -= nums[i];
                i++;
            }
            j++;
        }
        return res;
    }

    public int minSubArrayLen1(int s, int[] nums) {
        int n = nums.length;
        if (n == 0) {
            return 0;
        }
        int ans = Integer.MAX_VALUE;
        int start = 0, end = 0;
        int sum = 0;
        while (end < n) {
            sum += nums[end];
            while (sum >= s) {
                ans = Math.min(ans, end - start + 1);
                sum -= nums[start];
                start++;
            }
            end++;
        }
        return ans == Integer.MAX_VALUE ? 0 : ans;
    }

    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    //    283. 移动零
    public void moveZeroes(int[] nums) {
        if (nums == null || nums.length < 1)
            return;
        int len = nums.length;
        int i = 0, j = 0, count = 0;
        while (j < len) {
            if (nums[j] != 0) {
                swap(nums, i, j);
                i++;
            }
            j++;

        }
    }

    @Test
    public void test283() {
        int nums[] = new int[]{4, 5, 6, 7, 0, 1, 2};
        moveZeroes(nums);

        System.out.println(Arrays.toString(nums));
    }


    //498 对角线遍历
    public int[] findDiagonalOrder(int[][] mat) {
        if (mat == null)
            return null;

        int[] res = new int[mat.length * mat[0].length];
        int index = 0;

        for (int i = 0; i < mat.length; i++) {

            int row = i, column = i;
            int[] temp = new int[i + 1];
            int count = 0;
            while (row < mat.length && row >= 0 && column >= 0 && column < mat[0].length) {
                temp[count] = mat[row][column];
                row--;
                column++;
                count++;
            }


            if (row % 2 == 0) {
                count = 0;
                while (count < temp.length) {
                    res[index] = temp[count];
                    index++;
                    count++;
                }
            } else {
                count = temp.length - 1;
                while (count >= 0) {
                    res[index] = temp[count];
                    index++;
                    count--;
                }
            }
        }
        return res;
    }

    class CQueue {
        // 1,2,3
        Deque<Integer> stack1;
        Deque<Integer> stack2;

        public CQueue() {
            stack1 = new ArrayDeque<>();
            stack2 = new ArrayDeque<>();
        }

        public void appendTail(int value) {
            stack1.push(value);
        }

        public int deleteHead() {
            change();
            if (stack2.isEmpty())
                return -1;
            else
                return stack2.pop();
        }

        public void change() {
            if (stack2.isEmpty()) {
                while (!stack1.isEmpty()) {
                    stack2.push(stack1.poll());
                }
            }
        }
    }


    //    剑指 Offer 54. 二叉搜索树的第k大节点
    public int kthLargest1(TreeNode root, int k) {

        ArrayList<Integer> res = new ArrayList<>();
        dfs54(root, res);

        int res1 = -1;

        for (int i = 0; i < res.size() + 1 - k; i++) {
            res1 = res.get(i);
        }
        System.out.println(res);
        return res1;

    }

    public void dfs54(TreeNode treeNode, ArrayList<Integer> res) {

        if (treeNode == null)
            return;


        dfs54(treeNode.left, res);
        res.add(treeNode.val);
        dfs54(treeNode.right, res);

    }

    //    958. 二叉树的完全性检验


    public boolean isCompleteTree(TreeNode root) {
        List<aNode> nodes = new ArrayList<>();
        nodes.add(new aNode(root, 1));
        int i = 0;
        while (i < nodes.size()) {
            aNode anode = nodes.get(i++);
            if (anode.node != null) {
                nodes.add(new aNode(anode.node.left, anode.code * 2));
                nodes.add(new aNode(anode.node.right, anode.code * 2 + 1));
            }

        }

        return nodes.get(i - 1).code == nodes.size();
    }


    class aNode {
        TreeNode node;
        int code;

        aNode(TreeNode node, int code) {
            this.node = node;
            this.code = code;
        }
    }

    class Node1 {
        public int val;
        public Node1 left;
        public Node1 right;

        public Node1() {
        }

        public Node1(int _val) {
            val = _val;
        }

        public Node1(int _val, Node1 _left, Node1 _right) {
            val = _val;
            left = _left;
            right = _right;
        }
    }


    Node1 pre, head;

    public Node1 treeToDoublyList(Node1 root) {
        if (root == null)
            return null;
        dfs36(root);
        head.left = pre;
        pre.right = head;
        return head;
    }

    public void dfs36(Node1 cur) {

        if (cur == null)
            return;
        dfs36(cur.left);

        if (pre != null) pre.right = cur;
        else head = cur;
        cur.left = pre;
        pre = cur;


        dfs36(cur.right);
    }

    // 402. 移掉 K 位数字
    public String removeKdigits(String num, int k) {
        if (num == null || num.length() <= k)
            return "0";

        Deque<Character> stack = new ArrayDeque<>();

        for (int i = 0; i < num.length(); i++) {
            if (stack.isEmpty()) {
                stack.push(num.charAt(i));
            } else {
                while (!stack.isEmpty() && stack.peek() > num.charAt(i) && k > 0) {
                    stack.pop();
                    k--;
                }
                stack.push(num.charAt(i));
            }
        }
        while (k != 0) {
            stack.pop();
            k--;
        }


        String res = "";

        while (!stack.isEmpty()) {
            res = stack.pop() + res;
        }


        if (res.length() != 0)
            res = res.replaceFirst("^0*", "");
        return res.equals("") ? "0" : res;

    }


    //    139. 单词拆分
    Boolean[] dp;

    public Boolean wordBreak(String s, List<String> wordDict) {
        if (s == null || wordDict == null)
            return false;

        if (wordDict.contains(s))
            return true;

        dp = new Boolean[s.length() + 1];
        dp[0] = true;
        dfs139(s, wordDict);

        return dp[s.length()];


    }

    public Boolean dfs139(String s, List<String> wordDict) {

        boolean res = false;

        if (dp[s.length()] != null)
            return dp[s.length()];


        if (s.length() < 1)
            return true;

        for (String word : wordDict) {
            int len = word.length();
            if (s.length() >= len && wordDict.contains(s.substring(s.length() - len, s.length()))) {
                res = dfs139(s.substring(0, s.length() - len), wordDict) || res;
            }

        }

        dp[s.length()] = res;
        return res;
    }

    //    剑指 Offer 10- II. 青蛙跳台阶问题
    public int numWays(int n) {
        if (n <= 0)
            return 0;

        if (n == 1)
            return 1;

        if (n == 2)
            return 2;

        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        System.out.println();

        return dp[n];
    }

    List<Integer> res;

    public List<Integer> postorderTraversal(TreeNode root) {
        res = new ArrayList<>();

        if (root == null)
            return res;

        dfs145(root);
        return res;
    }

    public void dfs145(TreeNode root) {

        if (root == null)
            return;

        dfs145(root.left);
        dfs145(root.right);
        res.add(root.val);
    }


    //    207. 课程表

    List<List<Integer>> edges;
    int[] indeg;

    public boolean canFinish(int numCourses, int[][] prerequisites) {

        edges = new ArrayList<List<Integer>>();
        for (int i = 0; i < numCourses; i++) {
            edges.add(new ArrayList<Integer>());
        }
        indeg = new int[numCourses];


        for (int[] info : prerequisites) {
            edges.get(info[1]).add(info[0]);
            ++indeg[info[0]];
        }

        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indeg[i] == 0) {
                queue.offer(i);
            }
        }

        int visited = 0;
        while (queue.isEmpty()) {
            ++visited;
            int u = queue.poll();
            for (int v : edges.get(u)) {
                --indeg[v];
                if (indeg[v] == 0)
                    queue.offer(v);
            }

        }

        return visited == numCourses;

    }


    //739. 每日温度
    public int[] dailyTemperatures(int[] temperatures) {
        if (temperatures == null || temperatures.length < 1)
            return null;

        int[] res = new int[temperatures.length];

        Deque<Integer> stack = new ArrayDeque<>();

        for (int i = res.length - 1; i >= 0; i--) {
            if (stack.isEmpty()) {
                res[i] = 0;
                stack.push(i);
            } else {
                while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                    stack.pop();
                }
                if (stack.isEmpty()) {
                    res[i] = 0;
                } else
                    res[i] = stack.peek() - i;
                stack.push(i);
            }


        }

        return res;
    }

    @Test
    public void test739() {
        int nums[] = new int[]{4, 3, 6, 7, 0, 1, 2};
        int[] res = dailyTemperatures(nums);
        System.out.println(Arrays.toString(res));
    }


    //560. 和为 K 的子数组
    public int subarraySum(int[] nums, int k) {

        int res = 0;
        if (nums == null || nums.length < 1)
            return res;

        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);

        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if (map.containsKey(sum - k)) {
                res += map.get(sum - k);
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return res;
    }

    @Test
    public void test560() {
        int nums[] = new int[]{1, 0, 1, 0, 1};
        int res = subarraySum(nums, 2);
        System.out.println(res);
    }

    //518. 零钱兑换 II
    /*
    * 输入：amount = 5, coins = [1, 2, 5]
    输出：4
    解释：有四种方式可以凑成总金额：
    5=5
    5=2+2+1
    5=2+1+1+1
    5=1+1+1+1+1
    * */


    public int change(int amount, int[] coins) {
        int res = 0;
        if (coins == null || amount < 0)
            return res;

        int[] dp = new int[amount + 1];
        dp[0] = 1;


        for (int coin : coins) {
            for (int i = 1; i <= amount; i++) {
                if (i >= coin) {
                    dp[i] += dp[i - coin];
                }

            }
        }
        return dp[amount];
    }


    @Test
    public void test518() {
        int nums[] = new int[]{1, 2, 5};
        String s = "wete";
        int res = change(5, nums);
        System.out.println(res);
    }


    //79. 单词搜索


    public boolean exist(char[][] board, String word) {
        if (board == null || word == null || word.length() < 1)
            return false;

        boolean res = false;
        int row = board.length;
        int columns = board[0].length;

//        int[][] isVisited = new int[row][columns];

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < row; j++) {
                if (board[i][j] == word.charAt(0)) {
                    if (bfs79(i, j, board, word, new int[row][columns]))
                        return true;
                }
            }
        }
        return false;
    }

    boolean bfs79(int i, int j, char[][] board, String word, int[][] isVisited) {

        word = word.substring(1);
        if (word.equals(""))
            return true;

        boolean res = false;

        isVisited[i][j] = 1;
        System.out.println();

        for (int[] dir : directions) {
            int row = i + dir[0];
            int column = j + dir[1];
            if (row >= 0 && row < board.length && column >= 0 & column < board[0].length && (isVisited[row][column] == 0)) {
                if (board[row][column] == word.charAt(0)) {
                    res = bfs79(row, column, board, word, isVisited) || res;
                    if (res)
                        return true;
                }
            }
        }
        isVisited[i][j] = 0;
        return res;
    }

    @Test
    public void test76() {
        char nums[][] = new char[][]{{'A', 'B', 'C', 'E'}, {'S', 'F', 'C', 'S'}, {'A', 'D', 'E', 'E'}};
        String s = "ABCCED";
        boolean res = exist(nums, s);
        System.out.println(res);
    }

    //74 搜索二维矩阵
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null)
            return false;

        int rows = matrix.length;
        int columns = matrix[0].length;
        int i = 0, j = columns - 1;

        while (i < rows && j >= 0) {
            if (matrix[i][j] == target)
                return true;
            if (matrix[i][j] > target)
                j--;
            else
                i++;
        }
        return false;
    }

    public int fib(int n) {
        if (n <= 0)
            return 0;
        if (n == 1)
            return 1;

        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;

        for (int i = 2; i <= n; i++) {
            dp[i] = (dp[i - 1] + dp[i - 2]) % 1000000007;
        }
        return dp[n];


    }


    //59 螺旋矩阵11
    public int[][] generateMatrix(int n) {
        if (n <= 0)
            return null;

        int[][] res = new int[n][n];
        int i = 0, j = 0;
        int num = 1;
        int left = 0;
        int right = n - 1;
        int up = 0;
        int bottom = n - 1;

        while (true) {
            if (left > right)
                break;
            for (; j <= right; j++) {
                res[i][j] = num;
                num++;
            }
            up++;
            i++;
            j--;

            if (up > bottom)
                break;
            for (; i <= bottom; i++) {
                res[i][j] = num;
                num++;
            }
            right--;
            j--;
            i--;

            if (left > right)
                break;
            for (; j >= left; j--) {
                res[i][j] = num;
                num++;
            }
            bottom--;
            i--;
            j++;

            if (up > bottom)
                break;
            for (; i >= up; i--) {
                res[i][j] = num;
                num++;
            }
            j++;
            left++;
            i++;
        }

        return res;
//        System.out.println();

    }

    @Test
    public void test59() {

        int[][] res = generateMatrix(3);
        System.out.println(res);
    }

    // 61 旋转链表
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null || k < 1)
            return head;

        int length = 0;
        ListNode node = head;
        while (node != null) {
            length += 1;
            node = node.next;
        }

        if (length == 1)
            return head;

        k = k % length;

        int pre = length - k;

        node = head;
        for (int i = 0; i < k - 1; i++) {
            node = node.next;
        }

        ListNode newHead = node.next;
        node.next = null;

        ListNode temp = newHead;
        while (temp.next != null) {
            temp = temp.next;
        }
        temp.next = head;

        return newHead;
    }

    //剑指offer62  圆圈中最后剩下的数字
    //    1 2 3 4 5
    public int lastRemaining(int n, int m) {
        ArrayList<Integer> list = new ArrayList<>(n);

        for (int i = 0; i < n; i++) {
            list.add(i);
        }

        int idx = 0;
        while (n > 1) {
            idx = (idx + m - 1) % n;
            list.remove(idx);
            n--;
        }
        return list.get(0);
    }

    //剑指 Offer 40. 最小的k个数
    public int[] getLeastNumbers1(int[] arr, int k) {
        if (arr == null || arr.length < 1 || k < 1) {
            return null;
        }

        int[] res = new int[k];

        PriorityQueue<Integer> queue = new PriorityQueue<>((a, b) -> b - a);

        for (int num : arr) {
            if (queue.isEmpty() || queue.size() < k) {
                queue.offer(num);
            } else if (queue.peek() > num) {
                queue.offer(num);
                queue.poll();
            }
        }
        int i = 0;
        while (!queue.isEmpty()) {
            res[i] = queue.poll();
            i++;
        }
        return res;
    }

    public int[] getLeastNumbers(int[] arr, int k) {
        int[] res = new int[k];
        if (arr == null || arr.length < 1 || k < 1) {
            return res;
        }
//        return null;
        int i = 0, j = arr.length - 1;
        while (i != k) {
            int part = partition(arr, i, j);
            if (part == k) {
                i = k;
                break;
            }


            if (part > k) {
                j = part - 1;
            } else {
                i = part + 1;
            }
        }

        for (int m = 0; m < k; m++) {
            res[m] = arr[m];
        }
        return res;

    }


    public int partition1(int[] arr, int start, int end) {
        int part = end;
        int i = start, j = end;

        while (i < j) {
            while (arr[j] >= arr[part] && j > i) {
                j--;
            }
            while (arr[i] < arr[part] && i < j) {
                i++;
            }
            swap(arr, i, j);
        }
        swap(arr, i, part);
        return i;
    }

    int partition(int nums[], int i, int j) {

        int pivot = nums[j];
        int m = j;
        while (i < j) {
            while (nums[i] < pivot && i < j)
                i++;
            while (nums[j] >= pivot && i < j)
                j--;

            int tem = nums[i];
            nums[i] = nums[j];
            nums[j] = tem;
        }
        int tmp = nums[m];
        nums[m] = nums[i];
        nums[i] = pivot;

        return i;
    }


    @Test
    public void test40() {
        int[] ten = new int[]{3, 2, 1};
        int res1 = partition1(ten, 0, ten.length - 1);


        int[] res = getLeastNumbers(ten, 2);
        System.out.println(Arrays.toString(res));
//        System.out.println(res);

    }


    //jianzhi offer 21
    public int[] exchange(int[] nums) {
        if (nums == null || nums.length < 2)
            return nums;

        int i = 0, j = nums.length - 1;
        while (i < j) {

            while ((nums[j] & 1) == 0 && i < j) {
                j--;
            }

            while ((nums[i] & 1) == 1 && i < j) {
                i++;
            }
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
        }

        return nums;

    }


    public int kthSmallest(TreeNode root, int k) {
        int res = 0;
        if (root == null)
            return res;
        dfs230(root, k, res);
        return res;

    }

    public int dfs230(TreeNode root, int k, int res) {

        if (root == null)
            return 0;
        if (root.left != null)
            dfs230(root.left, k, res);

        if (k == 1) {
            res = root.val;
            return res;
        } else
            k--;

        if (root.right != null)
            dfs230(root.right, k, res);

        return res;
    }

    //135. 分发糖果
    public int candy(int[] ratings) {
        if (ratings == null || ratings.length < 1)
            return 0;

        int res = 0;
        int[] left = new int[ratings.length];
        int[] right = new int[ratings.length];
        Arrays.fill(left, 1);
        Arrays.fill(right, 1);

        for (int i = 1; i < ratings.length; i++) {
            if (ratings[i] > ratings[i - 1])
                left[i] = left[i - 1] + 1;
        }

        for (int i = ratings.length - 1; i >= 0; i--) {
            if (i > 0 && ratings[i - 1] > ratings[i])
                right[i - 1] = right[i] + 1;
            res += Math.max(right[i], left[i]);
        }

        return res;
    }

    public int maxProfitDP(int[] prices) {
        if (prices == null || prices.length <= 1) return 0;
        int[][][] dp = new int[prices.length][2][3];
        int MIN_VALUE = Integer.MIN_VALUE / 2;//因为最小值再减去1就是最大值Integer.MIN_VALUE-1=Integer.MAX_VALUE
        //初始化
        dp[0][0][0] = 0;//第一天休息
        dp[0][0][1] = dp[0][1][1] = MIN_VALUE;//不可能
        dp[0][0][2] = dp[0][1][2] = MIN_VALUE;//不可能
        dp[0][1][0] = -prices[0];//买股票
        for (int i = 1; i < prices.length; i++) {
            dp[i][0][0] = 0;
            dp[i][0][1] = Math.max(dp[i - 1][1][0] + prices[i], dp[i - 1][0][1]);
            dp[i][0][2] = Math.max(dp[i - 1][1][1] + prices[i], dp[i - 1][0][2]);
            dp[i][1][0] = Math.max(dp[i - 1][0][0] - prices[i], dp[i - 1][1][0]);
            dp[i][1][1] = Math.max(dp[i - 1][0][1] - prices[i], dp[i - 1][1][1]);
            dp[i][1][2] = MIN_VALUE;
        }
        return Math.max(0, Math.max(dp[prices.length - 1][0][1], dp[prices.length - 1][0][2]));
    }



}