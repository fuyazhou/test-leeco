import org.junit.Test;

import java.util.*;

public class leetcode {

    /*
    no.344
    编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 s 的形式给出。

    不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。

    示例 1：

    输入：s = ["h","e","l","l","o"]
    输出：["o","l","l","e","h"]
    示例 2：

    输入：s = ["H","a","n","n","a","h"]
    输出：["h","a","n","n","a","H"]

    来源：力扣（LeetCode）
    链接：https://leetcode-cn.com/problems/reverse-string
    著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    */

    public String[] reverseString(String[] string) {
        if (string == null)
            return null;
        int i = 0;
        int j = string.length - 1;
        while (i <= j) {

            String tem = string[i];
            string[i] = string[j];
            string[j] = tem;
            i++;
            j--;
        }
        return string;
    }

    @Test
    public void reverseStringTest() {
        System.out.println("reverse string test");
        String[] s = {"h", "e", "l", "l", "o"};
        System.out.println(Arrays.toString(s));
        String[] result = reverseString(s);
        System.out.println(Arrays.toString(result));
    }


/*      26. 删除有序数组中的重复项
        给你一个 升序排列 的数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。

        由于在某些语言中不能改变数组的长度，所以必须将结果放在数组nums的第一部分。更规范地说，如果在删除重复项之后有 k 个元素，那么 nums 的前 k 个元素应该保存最终结果。

        将最终结果插入 nums 的前 k 个位置后返回 k 。

        不要使用额外的空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

        示例 1：

        输入：nums = [1,1,2]
        输出：2, nums = [1,2,_]
        解释：函数应该返回新的长度 2 ，并且原数组 nums 的前两个元素被修改为 1, 2 。不需要考虑数组中超出新长度后面的元素。

        来源：力扣（LeetCode）
        链接：https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array
        著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
        */

    //    nums = [0,0,1,1,1,2,2,3,3,4]
    public int removeDuplicates(int[] nums) {

        if (nums == null)
            return 0;
        int i = 0;
        int n = 1;
        for (int j = 0; j < nums.length; j++) {
            if (nums[i] == nums[j]) {
                continue;
            } else {
                i++;
                n++;
                nums[i] = nums[j];
            }
        }
        return n;

    }

    @Test
    public void removeDuplicatesTest() {

        int[] nums = new int[]{0, 0, 1, 1, 1, 2, 2, 3, 3, 4};
        System.out.println(removeDuplicates(nums));
        System.out.println(Arrays.toString(nums));
    }


    /*
    给定字符串 S，找出最长重复子串的长度。如果不存在重复子串就返回 0。
    示例 1：
    输入："abcd"
    输出：0
    解释：没有重复子串。

    示例 2：
    输入："abbaba"
    输出：2
    解释：最长的重复子串为 "ab" 和 "ba"，每个出现 2 次。

    示例 3：
    输入："aabcaabdaab"
    输出：3
    解释：最长的重复子串为 "aab"，出现 3 次。

    示例 4：
    输入："aaaaa"
    输出：4
    解释：最长的重复子串为 "aaaa"，出现 2 次。

    提示：
    字符串 S 仅包含从 'a' 到 'z' 的小写英文字母。
            1 <= S.length <= 1500
    */

    // "aabcaabdaab"  --> 3
    public boolean isRepeatSubstring(String s, int mid) {
        Set<String> stringSet = new HashSet<>();

        for (int i = 0; i < s.length() - mid; i++) {
            if (stringSet.contains(s.substring(i, i + mid)))
                return true;
            stringSet.add(s.substring(i, i + mid));
        }
        return false;
    }

    public int longestSubstring(String s) {

        if (s == null || s.length() < 2)
            return 0;
        int i = 0, j = s.length() - 1;
        int mid = 0;
        int n = 0;
        String nn = "sa";


        while (i < j) {
            mid = i + (j - i + 1) / 2;
            if (isRepeatSubstring(s, mid)) {
                i = mid;
                n = mid;
            } else
                j = mid - 1;
        }

        return i;
    }

    @Test
    public void longestSubstringTest() {
        String s = "aaaaa";
        System.out.println(longestSubstring(s));

    }


    //链表相关问题
    class ListNode {
        int val;
        ListNode next;
    }

    //NO.876. 链表的中间结点
    public ListNode linkedListMiddleNode(ListNode head) {
        ListNode i = head, j = head;

        while (j != null && j.next != null) {
            i = i.next;
            j = j.next.next;
        }

        return i;
    }

    @Test
    public void linkedListMiddleNodeTest() {
        int[] a = {1, 2, 3, 4, 5};

    }

    //剑指 Offer 22. 链表中倒数第k个节点
    public ListNode getKthFromEnd(ListNode head, int k) {

        if (k < 1)
            return null;
        ListNode i = head, j = head;
        for (int m = 0; m < k; m++) {
            if (j == null)
                return null;
            j = j.next;
        }

        while (j != null) {
            i = i.next;
            j = j.next;
        }
        return i;
    }


    //no.206 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
    public ListNode reverseList(ListNode head) {

        ListNode i = head;
        ListNode pre = null;

        while (i != null) {
            ListNode temp = i.next;
            i.next = pre;
            pre = i;
            i = temp;
        }
        return pre;
    }

    public ListNode reverseListRecursion(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode headReverse = reverseListRecursion(head.next);
        head.next.next = head;
        head.next = null;
        return headReverse;
    }


//    739. 每日温度
//    给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指在第 i 天之后，才会有更高的温度。如果气温在这之后都不会升高，请在该位置用 0 来代替。
//
//
//
//    示例 1:
//
//    输入: temperatures = [73,74,75,71,69,72,76,73]
//    输出: [1,1,4,2,1,1,0,0]
//    示例 2:
//
//    输入: temperatures = [30,40,50,60]
//    输出: [1,1,1,0]
//    示例 3:
//
//    输入: temperatures = [30,60,90]
//    输出: [1,1,0]

    public int[] dailyTemperatures(int[] temperatures) {

        if (temperatures == null)
            return null;

        int[] result = new int[temperatures.length];
        //定义stack 栈   后进先出
        Deque<Integer> stack = new ArrayDeque();


        for (int i = temperatures.length - 1; i >= 0; i--) {
            int temperature = temperatures[i];

            if (stack.size() > 0) {
                if (temperature < temperatures[stack.peek()]) {
                    stack.push(i);

                    result[i] = 1;
                } else {
                    while (stack.size() > 0 && temperature >= temperatures[stack.peek()]) {
                        stack.pop();
                    }
                    if (stack.size() == 0) {
                        stack.push(i);
                        result[i] = 0;
                    } else {
                        result[i] = stack.peek() - i;
                        stack.push(i);
                    }
                }
            } else {
                stack.push(i);
                result[i] = 0;
            }
        }
        return result;
    }

    @Test
    public void dailyTemperaturesTest() {
        int[] t = new int[]{73, 74, 75, 71, 69, 72, 76, 73};
        System.out.println(Arrays.toString(dailyTemperatures(t)));
    }

/*

    735. 行星碰撞
    给定一个整数数组 asteroids，表示在同一行的行星。

    对于数组中的每一个元素，其绝对值表示行星的大小，正负表示行星的移动方向（正表示向右移动，负表示向左移动）。每一颗行星以相同的速度移动。

    找出碰撞后剩下的所有行星。碰撞规则：两个行星相互碰撞，较小的行星会爆炸。如果两颗行星大小相同，则两颗行星都会爆炸。两颗移动方向相同的行星，永远不会发生碰撞。

    示例 1：

    输入：asteroids = [5,10,-5]
    输出：[5,10]
    解释：10 和 -5 碰撞后只剩下 10 。 5 和 10 永远不会发生碰撞。
    示例 2：

    输入：asteroids = [8,-8]
    输出：[]
    解释：8 和 -8 碰撞后，两者都发生爆炸。
*/

    public int[] asteroidCollision(int[] asteroids) {
        //新建栈  stack
        Deque<Integer> stack = new ArrayDeque<>();
        for (int ast : asteroids) {
            if (ast > 0) {
                stack.push(ast);
            } else {
                while (!stack.isEmpty() && stack.peek() > 0 && stack.peek() < -ast) {
                    stack.pop();  //出栈
                }
                if (!stack.isEmpty() && stack.peek() == -ast) {
                    stack.pop();
                } else if (stack.isEmpty() || stack.peek() < 0) {
                    stack.push(ast);  //入栈
                }
            }
        }
        int[] res = new int[stack.size()];
        for (int i = res.length - 1; i >= 0; i--) {
            res[i] = stack.pop();
        }
        return res;


    }


    //  heap 堆的应用
    // 最大堆，最小堆
    // top k 的问题都是用heap实现
    /*
    215. 数组中的第K个最大元素
    给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。

    请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。




    示例 1:

    输入: [3,2,1,5,6,4] 和 k = 2
    输出: 5
    */
    public int findKthLargest(int[] nums, int k) {

        // 最小堆
        PriorityQueue<Integer> heap = new PriorityQueue<>();

        //最大堆
        PriorityQueue<Integer> heapMax = new PriorityQueue<>((a, b) -> {
            return b - a;
        });

        PriorityQueue<Integer> heapMax1 = new PriorityQueue<>((a, b) -> b - a);


        if (nums == null)
            return 0;

        for (int i = 0; i < nums.length; i++) {

            if (heap.size() < k || nums[i] > heap.peek()) {
                heap.offer(nums[i]);   //入堆
            }
            while (heap.size() > k)
                heap.poll();   //出堆
        }
        return heap.peek();
    }

    @Test
    public void findKthLargestTest() {
        int[] nums = new int[]{2, 1};
        int k = 2;

        System.out.println(findKthLargest(nums, k));

    }


    //    23. 合并K个升序链表
//    给你一个链表数组，每个链表都已经按升序排列。
//
//    请你将所有链表合并到一个升序链表中，返回合并后的链表。
//
//
//
//    示例 1：
//
//    输入：lists = [[1,4,5],[1,3,4],[2,6]]
//    输出：[1,1,2,3,4,4,5,6]
//    解释：链表数组如下：
//            [
//            1->4->5,
//            1->3->4,
//            2->6
//            ]
//    将它们合并到一个有序链表中得到。
//            1->1->2->3->4->4->5->6
    public ListNode mergeKLists(ListNode[] lists) {

        //最小堆
        PriorityQueue<ListNode> heap = new PriorityQueue<>((a, b) -> a.val - b.val);
        for (ListNode listNode : lists) {
            if (listNode != null)
                heap.offer(listNode);
        }
        ListNode res = new ListNode();
        ListNode cur = res;

        while (!heap.isEmpty()) {
            ListNode temp = heap.poll();
            cur.next = temp;
            cur = cur.next;
            if (temp.next != null)
                heap.offer(temp.next);
        }

        return res.next;
    }

//    1. 两数之和
//    给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
//
//    你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
//
//    你可以按任意顺序返回答案。
//    示例 1：
//
//    输入：nums = [2,7,11,15], target = 9
//    输出：[0,1]
//    解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1]
//
//    来源：力扣（LeetCode）
//    链接：https://leetcode-cn.com/problems/two-sum
//    著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

    public int[] twoSum(int[] nums, int target) {
        if (nums == null)
            return new int[2];

        HashMap<Integer, Integer> hashMap = new HashMap<>();
        int[] result = new int[2];


        for (int i = 0; i < nums.length; i++) {

            if (hashMap.containsKey(target - nums[i])) {
                result[0] = hashMap.get(target - nums[i]);
                result[1] = i;
                return result;
            } else {
                hashMap.put(nums[i], i);
            }
        }
        return result;
    }


    //    560. 和为 K 的子数组
//    给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。
//
//
//
//    示例 1：
//    1 2 3
//
//    输入：nums = [1,1,1], k = 2
//    输出：2
    public int subArraySum(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int sum = 0, cnt = 0;
        for (int x : nums) {
            sum += x;
            if (map.containsKey(sum - k)) {
                cnt += map.get(sum - k);
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return cnt;
    }


    // tree bfs(breadth-first search) 宽度优先搜索
    class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
    }

    //    102. 二叉树的层序遍历
//    给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。
//
//
//
//    示例 1：
//
//
//    输入：root = [3,9,20,null,null,15,7]
//    输出：[[3],[9,20],[15,7]]
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null)
            return new ArrayList<>();

        List<List<Integer>> result = new ArrayList<>();


        // 队列
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            List<Integer> list = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode temp = queue.peek();
                queue.poll();
                list.add(temp.val);
                if (temp.left != null)
                    queue.offer(temp.left);
                if (temp.right != null)
                    queue.offer(temp.right);

            }
            result.add(list);
        }
        return result;
    }


    //dfs
    //前序遍历
    public void preOrderTraversal(TreeNode root) {
        if (root == null)
            return;

        System.out.println(root.val);
        if (root.left != null)
            preOrderTraversal(root.left);
        if (root.right != null)
            preOrderTraversal(root.right);
    }

    //中序遍历
    public void inOrderTraversal(TreeNode root) {
        if (root == null)
            return;
        if (root.left != null)
            inOrderTraversal(root.left);

        System.out.println(root.val);

        if (root.right != null)
            inOrderTraversal(root.right);
    }

    //后序遍历
    public void postOrderTraversal(TreeNode root) {
        if (root == null)
            return;
        if (root.left != null)
            postOrderTraversal(root.left);
        if (root.right != null)
            postOrderTraversal(root.right);
        System.out.println(root.val);
    }


    //104 树的最大深度(也可以使用bfs进行搜索)
    public int maxDepth(TreeNode root) {

        if (root == null)
            return 0;

        int right = maxDepth(root.right);
        int left = maxDepth(root.left);
        return Math.max(right, left) + 1;
    }

//    124. 二叉树中的最大路径和
//    路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。
//
//    路径和 是路径中各节点值的总和。
//
//    给你一个二叉树的根节点 root ，返回其 最大路径和 。
//
//
//
//    示例 1：
//
//
//    输入：root = [1,2,3]
//    输出：6
//    解释：最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6

    int max = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        dfs124(root);
        return max;

    }

    public int dfs124(TreeNode node) {
        if (node == null)
            return 0;
        int left = dfs124(node.left);
        int right = dfs124(node.right);
        left = left < 0 ? 0 : left;
        right = right < 0 ? 0 : right;
        max = Math.max(max, left + right + node.val);
        return Math.max(left + node.val, right + node.val);
    }

//    129. 求根节点到叶节点数字之和
//    给你一个二叉树的根节点 root ，树中每个节点都存放有一个 0 到 9 之间的数字。
//    每条从根节点到叶节点的路径都代表一个数字：
//
//    例如，从根节点到叶节点的路径 1 -> 2 -> 3 表示数字 123 。
//    计算从根节点到叶节点生成的 所有数字之和 。
//
//    叶节点 是指没有子节点的节点。

//    示例 1：
//    输入：root = [1,2,3]
//    输出：25
//    解释：
//    从根到叶子节点路径 1->2 代表数字 12
//    从根到叶子节点路径 1->3 代表数字 13
//    因此，数字总和 = 12 + 13 = 25

    int sum129 = 0;

    public int sumNumbers(TreeNode root) {
        if (root == null)
            return 0;
        dfs129(root, 0);
        return sum129;
    }

    public void dfs129(TreeNode node, int num) {
        num = num * 10 + node.val;
        if (node.left == null && node.right == null) {
            sum129 += num;
            return;
        }
        if (node.right != null)
            dfs129(node.right, num);
        if (node.left != null)
            dfs129(node.left, num);
    }

//
//    236. 二叉树的最近公共祖先
//    给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
//
//    百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
//
//
//
//    示例 1：
//
//
//    输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
//    输出：3
//    解释：节点 5 和节点 1 的最近公共祖先是节点 3 。

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {

        if (root == null)
            return null;
        if (root == p || root == q)
            return root;

        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);

        if (left == null)
            return right;
        if (right == null)
            return left;
        if (right != null && left != null)
            return root;
        return null;
    }

//    542. 01 矩阵
//    给定一个由 0 和 1 组成的矩阵 mat ，请输出一个大小相同的矩阵，其中每一个格子是 mat 中对应位置元素到最近的 0 的距离。
//
//    两个相邻元素间的距离为 1 。
//    示例 1：
//
//    输入：mat = [[0,0,0],[0,1,0],[0,0,0]]
//    输出：[[0,0,0],[0,1,0],[0,0,0]]

    int[][] directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    public int[][] updateMatrix(int[][] mat) {
        int m = mat.length;
        int n = mat[0].length;
        int[][] res = new int[m][n];
        Boolean[][] isVisited = new Boolean[m][n];
        for (int q = 0; q < m; q++) {
            for (int w = 0; w < n; w++) {
                isVisited[q][w] = false;
            }
        }

        Queue<int[]> queue = new LinkedList<>();

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (mat[i][j] == 0) {
//                    res[i][j] = 0;
                    isVisited[i][j] = true;
                    queue.offer(new int[]{i, j});
                }
            }
        }

        int level = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] index = queue.poll();
                if (mat[index[0]][index[1]] == 1) {
                    res[index[0]][index[1]] = level;
//                    isVisited[index[0]][index[1]]=true;
                }
                for (int[] dir : directions) {

                    int a = index[0] + dir[0];
                    int b = index[1] + dir[1];
                    if (a > 0 && a < m && b > 0 && b < n && isVisited[a][b] == false) {
                        isVisited[a][b] = true;
                        queue.offer(new int[]{a, b});
                    }

                }
            }
            level++;
        }

        return res;

    }

    @Test
    public void updateMatrixTest() {
        int[][] mat = new int[][]{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}};
        int[][] res = updateMatrix(mat);
        System.out.println(Arrays.toString(res));

    }

    //
//    127. 单词接龙
//    字典 wordList 中从单词 beginWord 和 endWord 的 转换序列 是一个按下述规格形成的序列 beginWord -> s1 -> s2 -> ... -> sk：
//
//    每一对相邻的单词只差一个字母。
//    对于 1 <= i <= k 时，每个 si 都在 wordList 中。注意， beginWord 不需要在 wordList 中。
//    sk == endWord
//    给你两个单词 beginWord 和 endWord 和一个字典 wordList ，返回 从 beginWord 到 endWord 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列，返回 0
//

    //首先构建图
    public Map<String, List<String>> constructGraph(List<String> wordList) {
        Map<String, List<String>> graph = new HashMap<>();

        for (int i = 0; i < wordList.size(); i++) {
            for (int j = i + 1; j < wordList.size(); j++) {
                if (isOneWayChange(wordList.get(i), wordList.get(j))) {
                    graph.computeIfAbsent(wordList.get(i), k -> new ArrayList<>()).add(wordList.get(j));
                    graph.computeIfAbsent(wordList.get(j), k -> new ArrayList<>()).add(wordList.get(i));
                }
            }
        }
        return graph;
    }

    //判断两个word是否通过能通过一步转换过来
    Boolean isOneWayChange(String word1, String word2) {
        int diff = 0;
        for (int i = 0; i < word1.length(); i++) {
            if (word1.charAt(i) != word2.charAt(i))
                diff++;
        }
        return diff == 1;

    }

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        if (!wordList.contains(endWord))
            return 0;
        if (!wordList.contains(beginWord))
            wordList.add(beginWord);

        Map<String, List<String>> graph = constructGraph(wordList);
        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        queue.offer(beginWord);

        int cost = 1;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                String word = queue.poll();
                if (word.equals(endWord))
                    return cost;
                List<String> subWords = graph.getOrDefault(word, new ArrayList<>());
                for (String subWord : subWords) {
                    if (!visited.contains(subWord)) {
                        visited.add(subWord);
                        queue.offer(subWord);
                    }
                }
            }
            cost++;
        }
        return 0;

    }


//    743. 网络延迟时间
//    有 n 个网络节点，标记为 1 到 n。
//
//    给你一个列表 times，表示信号经过 有向 边的传递时间。 times[i] = (ui, vi, wi)，其中 ui 是源节点，vi 是目标节点， wi 是一个信号从源节点传递到目标节点的时间。
//
//    现在，从某个节点 K 发出一个信号。需要多久才能使所有节点都收到信号？如果不能使所有节点收到信号，返回 -1 。
//
//
//
//    示例 1：
//
//
//
//    输入：times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
//    输出：2


    class Cell {
        int node, time;

        Cell(int node, int time) {
            this.node = node;
            this.time = time;
        }

    }

    public int networkDelayTime(int[][] times, int N, int k) {
        Map<Integer, List<Cell>> map = new HashMap<>();
        for (int[] time : times) {
            List<Cell> edges = map.getOrDefault(time[0], new ArrayList<>());
            edges.add(new Cell(time[1], time[2]));
            map.put(time[0], edges);
        }

        Map<Integer, Integer> costs = new HashMap<>();

        //最小堆
        PriorityQueue<Cell> heap = new PriorityQueue<>((a, b) -> a.time - b.time);


        heap.offer(new Cell(k, 0));

        while (!heap.isEmpty()) {
            Cell cur = heap.poll();

            if (costs.containsKey(cur.node))
                continue;

            costs.put(cur.node, cur.time);
            if (map.containsKey(cur.node)) {
                for (Cell nei : map.get(cur.node)) {
                    if (!costs.containsKey(nei.node)) {
                        heap.offer(new Cell(nei.node, cur.time + nei.time));
                    }
                }
            }
        }
        if (costs.size() != N)
            return -1;

        int res = 0;

        for (int x : costs.values())
            res = Math.max(res, x);
        return res;
    }
//
//    787. K 站中转内最便宜的航班
//    有 n 个城市通过一些航班连接。给你一个数组 flights ，其中 flights[i] = [fromi, toi, pricei] ，表示该航班都从城市 fromi 开始，以价格 pricei 抵达 toi。
//
//    现在给定所有的城市和航班，以及出发城市 src 和目的地 dst，你的任务是找到出一条最多经过 k 站中转的路线，使得从 src 到 dst 的 价格最便宜 ，并返回该价格。 如果不存在这样的路线，则输出 -1。

    class Cell1 {
        int dst, stop, price;

        public Cell1(int dst, int stop, int price) {
            this.dst = dst;
            this.stop = stop;
            this.price = price;
        }

    }

    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {

        Map<Integer, List<int[]>> map = new HashMap<>();

        for (int[] fight : flights) {
            List<int[]> to = map.getOrDefault(fight[0], new ArrayList<>());
            to.add(new int[]{fight[1], fight[2]});
            map.put(fight[0], to);
        }

        PriorityQueue<Cell1> heap = new PriorityQueue<>((a, b) -> a.price - b.price);
        heap.offer(new Cell1(src, k, 0));

        while (!heap.isEmpty()) {
            Cell1 cur = heap.poll();
            if (cur.dst == dst) {
                return cur.price;
            }
            if (cur.stop >= 0 && map.containsKey(cur.dst)) {
                for (int[] next : map.get(cur.dst)) {
                    heap.offer(new Cell1(next[0], cur.stop - 1, cur.price + next[1]));
                }
            }
        }
        return -1;
    }

    @Test
    public void findCheapestPriceTest() {
        int n = 3, src = 0, dst = 2, k = 1;
        int[][] edges = {{0, 1, 100}, {1, 2, 100}, {0, 2, 500}};

        int m = findCheapestPrice(n, edges, src, dst, k);

        System.out.println(m);

    }


//    200. 岛屿数量
//    给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
//
//    岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
//
//    此外，你可以假设该网格的四条边均被水包围。

    //int[][] directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    public int numIslands(char[][] grid) {
        if (grid == null)
            return 0;
        int nums = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    nums += 1;
                    dfs200(grid, i, j);
                }
            }
        }
        return nums;

    }

    public void dfs200(char[][] grid, int i, int j) {
        if (grid[i][j] == '1') {
            grid[i][j] = '0';
            for (int[] direction : directions) {
                int m = direction[0];
                int n = direction[1];
                if (i + m >= 0 && i + m < grid.length && j + n >= 0 && j + n < grid[0].length) {
                    dfs200(grid, i + m, j + n);
                }
            }

        }
    }


//    332. 重新安排行程
//    给你一份航线列表 tickets ，其中 tickets[i] = [fromi, toi] 表示飞机出发和降落的机场地点。请你对该行程进行重新规划排序。
//
//    所有这些机票都属于一个从 JFK（肯尼迪国际机场）出发的先生，所以该行程必须从 JFK 开始。如果存在多种有效的行程，请你按字典排序返回最小的行程组合。
//
//    例如，行程 ["JFK", "LGA"] 与 ["JFK", "LGB"] 相比就更小，排序更靠前。
//    假定所有机票至少存在一种合理的行程。且所有的机票 必须都用一次 且 只能用一次。

    public List<String> findItinerary(List<List<String>> tickets) {

        Map<String, PriorityQueue<String>> map = new HashMap<>();

        for (List<String> edges : tickets) {
            map.computeIfAbsent(edges.get(0), k -> new PriorityQueue<String>()).offer(edges.get(1));
        }

        List<String> res = new LinkedList<>();
        dfs332(res, map, "JFK");
        return res;

    }


    public void dfs332(List<String> res, Map<String, PriorityQueue<String>> map, String cur) {
        PriorityQueue<String> neis = map.getOrDefault(cur, new PriorityQueue<>());
        while (!neis.isEmpty()) {
            dfs332(res, map, neis.poll());
        }
        res.add(0, cur);
    }

    @Test
    public void findItineraryTest() {

        String[][] m = new String[][]{{"MUC", "LHR"}, {"JFK", "MUC"}, {"SFO", "SJC"}, {"LHR", "SFO"}};
        List<List<String>> tickes = new ArrayList<>();
        for (int i = 0; i < m.length; i++) {
            tickes.add(Arrays.asList(m[i]));
        }

        List<String> mm = findItinerary(tickes);
        System.out.println(mm);


    }


}
