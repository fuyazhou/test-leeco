import org.junit.Test;

import java.util.*;

public class leetCodeByMe {


//    给定一个字符串 s ，请你找出其中不含有重复字符的最长子串的长度。
//
//    输入: s = "abcabcbb"
//    输出: 3
//    解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
//
//    来源：力扣（LeetCode）
//    链接：https://leetcode-cn.com/problems/longest-substring-without-repeating-characters
//    著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

    // 利用二分法做
    public Boolean isWithOutRepating(String m) {
        if (m.length() == 1)
            return true;
        HashSet<String> set = new HashSet<>();

        for (int i = 0; i < m.length(); i++) {
            if (set.contains(m.substring(i, i + 1)))
                return false;
            set.add(m.substring(i, i + 1));
        }
        return true;
    }

    public Boolean is(String s, int m) {

        for (int x = 0; x < s.length() - m + 1; x++) {
            if (isWithOutRepating(s.substring(x, x + m))) {
                return true;
            }
        }
        return false;
    }

    public int lengthOfLongestSubstring(String s) {
        if (s == null)
            return 0;
        if (s.length() == 1)
            return 1;

        int i = 0, j = s.length();
        while (i < j) {
            int m = (i + j + 1) / 2;
            if (is(s, m)) {
                i = m;
            } else
                j = m - 1;
        }
        return i;
    }


    public int lengthOfLongestSubstring1(String s) {
        if (s == null || s.length() == 0)
            return 0;

        HashMap<Character, Integer> hashMap = new HashMap<>();
        int res = 0;
        int left = 0;

        for (int i = 0; i < s.length(); i++) {

            if (hashMap.containsKey(s.charAt(i))) {
                left = Math.max(left, hashMap.get(s.charAt(i)) + 1);

            }
            hashMap.put(s.charAt(i), i);

            res = Math.max(res, i - left + 1);


        }
        return res;


    }


    @Test
    public void lengthOfLongestSubstringTest() {
        String s = "abcabcbb";
        int m = lengthOfLongestSubstring1(s);
        System.out.println(m);

    }


    //    53. 最大子数组和
//    给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
//
//    子数组 是数组中的一个连续部分。
//
//    示例 1：
//
//    输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
//    输出：6
//    解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
    public int maxSubArray(int[] nums) {

        if (nums == null)
            return 0;

        int res = Integer.MIN_VALUE;
//        int res = nums[0];

        int sum = 0;

        for (int i = 0; i < nums.length; i++) {
            sum = sum + nums[i];
            res = Math.max(res, sum);
            if (sum < 0)
                sum = 0;
        }
        return res;
    }


    //链表相关问题
    class ListNode {
        int val;
        ListNode next;

        public ListNode() {
        }

        public ListNode(int i, ListNode head) {
            val = i;
            head = next;
        }

        public ListNode(int i) {
            val = i;
        }
    }

    //    21. 合并两个有序链表
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {

        ListNode head = new ListNode();

        if (list1 == null)
            return list2;
        if (list2 == null)
            return list1;

        ListNode cur = head;


        while (list1 != null && list2 != null) {
            if (list1.val > list2.val) {
                cur.next = list2;
                list2 = list2.next;
            } else {
                cur.next = list1;
                list1 = list1.next;
            }
            cur = cur.next;
        }
        if (list1 == null)
            cur.next = list2;
        if (list2 == null)
            cur.next = list1;

        return head;

    }


    @Test
    public void maxSubArrayTest() {
        int[] nums = new int[]{-2, 1, -3, 4, -1, 2, 1, -5, 4};

        int res = maxSubArray(nums);

        System.out.println(res);

    }


    class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        public TreeNode(int root_val) {
            this.val = root_val;
        }
    }
//    103. 二叉树的锯齿形层序遍历
//    给你二叉树的根节点 root ，返回其节点值的 锯齿形层序遍历 。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {

        if (root == null)
            return null;

        List<List<Integer>> res = new ArrayList<>();

        Queue<TreeNode> queue = new LinkedList<>();
        int loop = 0;

        queue.offer(root);

        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> subRes = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode tem = queue.poll();
                if (tem.left != null) {
                    queue.offer(tem.left);
                }
                if (tem.right != null) {
                    queue.offer(tem.right);
                }

                subRes.add(tem.val);

            }
            loop++;

            if (loop % 2 == 0) {
                res.add(subRes);

            } else {
                Collections.reverse(subRes);
                res.add(subRes);
            }


        }
        return res;


    }

//    141. 环形链表
//    给你一个链表的头节点 head ，判断链表中是否有环。
//
//    如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。
//
//    如果链表中存在环 ，则返回 true 。 否则，返回 false 。

    public boolean hasCycle(ListNode head) {

        if (head == null)
            return false;
        ListNode i = head, j = head;
        while (i != null && j != null && j.next != null) {
            i = i.next;
            j = j.next.next;
            if (i == j)
                return true;
        }
        return false;
    }


    //    121. 买卖股票的最佳时机
    public int maxProfit(int[] prices) {

        if (prices == null)
            return 0;

        int res = 0;
        int min = Integer.MIN_VALUE;

        for (int i = 0; i < prices.length; i++) {

            min = Math.min(min, prices[i]);

            res = Math.max(res, prices[i] - min);

        }

        return res;
    }


    //20. 有效的括号
    public boolean isValid(String s) {

        Map<Character, Character> map = new HashMap<Character, Character>() {{
            put('{', '}');
            put('[', ']');
            put('(', ')');
            put('?', '?');
        }};

        Deque<Character> stack = new ArrayDeque<>();

        if (s == null || s.length() == 0)
            return true;

        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))) {
                stack.push(s.charAt(i));
            }
            if (map.containsValue(s.charAt(i))) {

                if (!stack.isEmpty()) {
                    Character tem = stack.pop();
                    if (!map.get(tem).equals(s.charAt(i))) {
                        return false;
                    }
                } else
                    return false;
            }
        }

        if (stack.isEmpty())
            return true;
        else
            return false;
    }


//    236. 二叉树的最近公共祖先

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null)
            return null;

        if (p == root || q == root)
            return root;

        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);

        if (left == null)
            return right;
        if (right == null)
            return left;
        if (left != null && right != null)
            return root;

        return null;
    }

    //    160. 相交链表
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {

        if (headA == null || headB == null)
            return null;
        int lenA = 0, lenB = 0;

        ListNode curA = headA;
        ListNode curB = headB;
        while (curA != null) {
            lenA++;
            curA = curA.next;
        }
        while (curB != null) {
            lenB++;
            curB = curB.next;
        }

        curA = headA;
        curB = headB;

        if (lenA > lenB) {
            for (int i = 0; i < lenA - lenB; i++) {
                curA = curA.next;
            }
        }
        if (lenB > lenA) {
            for (int i = 0; i < lenB - lenA; i++) {
                curB = curB.next;
            }
        }

        while (curA != null) {

            if (curA == curB)
                return curA;
            else {
                curA = curA.next;
                curB = curB.next;
            }
        }
        return null;
    }


//    88. 合并两个有序数组

    public void merge(int[] nums1, int m, int[] nums2, int n) {


        int length = nums1.length;

        while (m - 1 >= 0 && n - 1 >= 0) {
            if (nums1[m - 1] >= nums2[n - 1]) {
                nums1[length - 1] = nums1[m - 1];
                m--;
            } else {
                nums1[length - 1] = nums2[n - 1];
                n--;
            }
            length--;
        }
        if (m - 1 >= 0) {
            while (m - 1 >= 0) {
                nums1[length - 1] = nums1[m - 1];
                m--;
                length--;
            }
        }
        if (n - 1 >= 0) {
            while (n - 1 >= 0) {
                nums1[length - 1] = nums2[n - 1];
                n--;
                length--;
            }
        }
    }

    @Test
    public void mergeTest() {
        int[] nums1 = new int[]{1, 2, 3, 0, 0, 0};
        int m = 3;
        int[] nums2 = new int[]{2, 5, 6};
        int n = 3;
        merge(nums1, m, nums2, n);
    }


    //    142. 环形链表 II
    public ListNode detectCycle(ListNode head) {
        ListNode pos = head;

        Set<ListNode> visited = new HashSet<>();

        while (pos != null) {

            if (visited.contains(pos))
                return pos;
            else
                visited.add(pos);
            pos = pos.next;

        }
        return null;


    }

    //    92. 反转链表 II
    public ListNode reverseBetween(ListNode head, int left, int right) {

        Deque<ListNode> stack = new ArrayDeque<>();
        ListNode cur = head;


        int i = 1;
        while (i < left - 1) {
            cur = cur.next;
            i++;
        }
        ListNode temp = cur;

        temp = temp.next;
        stack.push(temp);

        int j = 0;
        while (j < right - left) {
            temp = temp.next;
            stack.push(temp);
            j++;
        }

        ListNode temp1 = temp.next;

        while (!stack.isEmpty()) {
            ListNode m = stack.pop();
            m.next = null;
            cur.next = m;
            cur = m;
        }
        cur.next = temp1;
        return head;
    }

    //    300. 最长递增子序列

    Integer[][] memo;

    public int lengthOfLIS(int[] nums) {
        if (nums.length == 0)
            return 0;


        int len = nums.length;
        int[] dp = new int[len];


        dp[0] = 1;
        int max = 1;

        for (int i = 1; i < nums.length; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[j] + 1, dp[i]);
                }
            }
            max = Math.max(max, dp[i]);
        }
        return max;
    }

    //    704. 二分查找
    //    输入: nums = [-1,0,3,5,9,12], target = 9
    //    输出: 4
    //    解释: 9 出现在 nums 中并且下标为
    public int search(int[] nums, int target) {
        if (nums == null || nums.length <= 0)
            return -1;


        int i = 0, j = nums.length - 1;

        while (i <= j) {
            int mid = i + (j - i + 1) / 2;

            if (target == nums[mid])
                return mid;

            if (nums[mid] > target)
                j = mid - 1;

            if (nums[mid] < target)
                i = mid + 1;
        }

        return -1;
    }


    @Test
    public void searchTest() {
        int[] nums = new int[]{-1, 0, 3, 5, 9, 12};
        int target = 2;
        int m = search(nums, target);

        System.out.println(m);

    }


    //    94. 二叉树的中序遍历
    List<Integer> mome94;

    public List<Integer> inorderTraversal(TreeNode root) {

        mome94 = new ArrayList<>();

        dfs94(root);

        return mome94;


    }

    public void dfs94(TreeNode root) {

        if (root == null)
            return;

        if (root.left != null)
            dfs94(root.left);
        mome94.add(root.val);
        if (root.right != null)
            dfs94(root.right);
    }

    //143. 重排链表
    public void reorderList(ListNode head) {

        if (head == null)
            return;

        Deque<ListNode> stack = new ArrayDeque<>();
        Queue<ListNode> queue = new LinkedList<>();

        int i = 0;

        ListNode cur = head;
        while (cur != null) {
            i++;
            cur = cur.next;
        }

        int mid = i / 2 + 1;

        int j = 0;
        cur = head;
        while (j < mid) {
            queue.offer(cur);
            cur = cur.next;
            j++;
        }
        while (j < i) {
            stack.push(cur);
            cur = cur.next;
            j++;
        }

        ListNode newHead = new ListNode();
        ListNode cur1 = newHead;
        while ((!stack.isEmpty()) || (!queue.isEmpty())) {

            if (!queue.isEmpty()) {
                ListNode temp = queue.poll();
                temp.next = null;
                newHead.next = temp;
                newHead = newHead.next;
            }
            if (!stack.isEmpty()) {
                ListNode temp = stack.pop();
                temp.next = null;
                newHead.next = temp;
                newHead = newHead.next;
            }
        }
        head = cur1.next;
    }

    //    199. 二叉树的右视图
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();

        if (root == null)
            return res;

        Queue<TreeNode> queue = new LinkedList<>();

        queue.offer(root);

        while (!queue.isEmpty()) {

            int size = queue.size();

            for (int i = 0; i < size; i++) {
                TreeNode temp = queue.poll();
                if (i == 0)
                    res.add(temp.val);

                if (temp.right != null)
                    queue.offer(temp.right);
                if (temp.left != null)
                    queue.offer(temp.left);
            }

        }
        return res;
    }

//    70. 爬楼梯

    public int climbStairs(int n) {

        int[] dp = new int[n + 1];
        if (n < 0)
            return 0;
        if (n == 1)
            return 1;
        if (n == 2)
            return 2;

        dp[0] = 0;
        dp[1] = 1;
        dp[2] = 2;
        if (n <= 2)
            return dp[n];

        for (int i = 3; i < n + 1; i++) {

            dp[i] = dp[i - 1] + dp[i - 2];

        }

        return dp[n];
    }

//    124. 二叉树中的最大路径和

    int max = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        dfs124(root);
        return max;
    }

    public int dfs124(TreeNode root) {

        if (root == null)
            return 0;

        int left = dfs124(root.left);
        int right = dfs124(root.right);

        left = left < 0 ? 0 : left;
        right = right < 0 ? 0 : right;


        max = Math.max(max, left + root.val + right);

        return Math.max(left + root.val, right + root.val);

    }

    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length < 2) return nums;
        // 双向队列 保存当前窗口最大值的数组位置 保证队列中数组位置的数值按从大到小排序
        LinkedList<Integer> queue = new LinkedList();

        // 结果数组
        int[] result = new int[nums.length - k + 1];
        // 遍历nums数组
        for (int i = 0; i < nums.length; i++) {
            // 保证从大到小 如果前面数小则需要依次弹出，直至满足要求
            while (!queue.isEmpty() && nums[queue.peekLast()] <= nums[i]) {
                queue.pollLast();
            }
            // 添加当前值对应的数组下标
            queue.addLast(i);
            // 判断当前队列中队首的值是否有效
            if (queue.peek() <= i - k) {
                queue.poll();
            }
            // 当窗口长度为k时 保存当前窗口中最大值
            if (i + 1 >= k) {
                result[i + 1 - k] = nums[queue.peek()];
            }
        }
        return result;
    }

    //    19. 删除链表的倒数第 N 个结点
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head);
        ListNode first = head;
        ListNode second = dummy;
        for (int i = 0; i < n; ++i) {
            first = first.next;
        }
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        second.next = second.next.next;
        ListNode ans = dummy.next;
        return ans;
    }

    //    69. x 的平方根
    public int mySqrt(int x) {
        int l = 0, r = x, ans = -1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if ((long) mid * mid <= x) {
                ans = mid;
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return ans;
    }

    @Test
    public void mySqrtTest() {

        int res = mySqrt(5);
        System.out.println(res);
    }

    //82. 删除排序链表中的重复元素 II
    public ListNode deleteDuplicates(ListNode head) {

        if (head == null)
            return null;

        Map<Integer, Boolean> map = new HashMap();

        ListNode quick = head, slow = head;

        while (quick.next != null) {
            quick = quick.next;
            if (quick.val == slow.val) {
                map.put(slow.val, true);
            }
            slow = slow.next;
        }
        ListNode newhaed = new ListNode(0, head);
        quick = newhaed;
        slow = newhaed;
        while (quick.next != null) {
            quick = quick.next;
            if (map.containsKey(quick.val)) {
                slow.next = quick.next;
                quick = slow;
            } else
                slow = slow.next;

        }

        return newhaed.next;

    }

    //    4. 寻找两个正序数组的中位数
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1 == null || nums2 == null)
            return 0;
        int len1 = nums1.length;
        int len2 = nums2.length;
        int len = len1 + len2;
        int i = 0, j = 0;
        int[] num = new int[len1 + len2];
        int m = 0;
        while (i < len1 || j < len2) {
            if (i >= len1) {
                num[m] = nums2[j];
                m++;
                j++;
            } else if (j >= len2) {
                num[m] = nums1[i];
                m++;
                i++;
            } else {
                if (nums1[i] > nums2[j]) {
                    num[m] = nums2[j];
                    m++;
                    j++;
                } else {
                    num[m] = nums1[i];
                    m++;
                    i++;
                }

            }
        }
        double res;
        if (len % 2 == 0) {
            res = (double) (num[len / 2] + num[len / 2 - 1]) / 2;
        } else {
            res = (double) num[len / 2];
        }
        return res;
    }

    @Test
    public void findMedianSortedArraysTest() {
        int[] nums1 = new int[]{1, 3};
        int[] nums2 = new int[]{2, 4};

//        double res = findMedianSortedArrays(nums1, nums2);
        String a = " f ";


        System.out.println(a.trim());
    }

    //    148. 排序链表
    public ListNode sortList(ListNode head) {
        if (head == null)
            return null;
        ListNode cur = head;
        List<Integer> value = new ArrayList<>();
        while (cur != null) {
            value.add(cur.val);
            cur = cur.next;
        }
        Collections.sort(value);
//        System.out.println(Arrays.asList(value));
        ListNode newHead = new ListNode(0, null);
        cur = newHead;
        for (Integer i : value) {
            ListNode tem = new ListNode(i, null);
            cur.next = tem;
            cur = cur.next;
        }
        return newHead.next;

    }

    //72,编辑距离
//    Integer[][] memo;
    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();

        memo = new Integer[m + 1][n + 1];

        for (int i = 0; i <= m; i++)
            memo[i][0] = i;
        for (int i = 0; i <= n; i++)
            memo[0][i] = i;

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i) == word2.charAt(j)) {
                    memo[i][j] = memo[i - 1][j - 1];
                } else {
                    memo[i][j] = Math.min(Math.min(memo[i - 1][j], memo[i][j - 1]), memo[i - 1][j - 1]) + 1;
                }


            }
        }
        return memo[m][n];
    }

    //    2. 两数相加
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null)
            return l2;
        if (l2 == null)
            return l1;

        ListNode head = new ListNode(0, null);
        ListNode head1 = new ListNode(0);
        ListNode cur = head;
        int carry = 0;
        int sum = 0;


        while (l1 != null || l2 != null) {
            if (l1 != null && l2 != null) {
                sum = l1.val + l2.val + carry;
                l1 = l1.next;
                l2 = l2.next;
            } else if (l1 == null) {
                sum = l2.val + carry;
                l2 = l2.next;
            } else {
                sum = l1.val + carry;
                l1 = l1.next;
            }

            carry = sum / 10;
            sum = sum % 10;

            ListNode temp = new ListNode(sum, null);
            cur.next = temp;
            cur = cur.next;
        }
        if (carry != 0) {
            ListNode temp = new ListNode(carry, null);
            cur.next = temp;
            cur = cur.next;
        }
        System.out.println();
        return head.next;
    }

    //22. 括号生成
    public List<String> generateParenthesis(int n) {
        if (n == 0)
            return null;

        List<String> res = new ArrayList<>();
        return res;


    }


    // 144. 二叉树的前序遍历
    List<Integer> res;

    public List<Integer> preorderTraversal(TreeNode root) {
        res = new ArrayList<>();

        dfs114(root);
        return res;

    }

    public void dfs114(TreeNode root) {
        if (root == null)
            return;
        res.add(root.val);
        if (root.left != null)
            dfs114(root.left);
        if (root.right != null)
            dfs114(root.right);
    }

    //    239. 滑动窗口最大值
    public int[] maxSlidingWindow1(int[] nums, int k) {
        if (nums == null || nums.length < 2)
            return nums;

        int[] res = new int[nums.length - k + 1];

        LinkedList<Integer> queue = new LinkedList<>();

        for (int i = 0; i < nums.length; i++) {

            if (!queue.isEmpty() && nums[queue.peekLast()] < nums[i]) {
                queue.pollLast();
            }
            queue.addLast(i);

            if (queue.peek() <= i - k) {
                queue.poll();
            }
            if (i + 1 >= k) {
                res[i - k + 1] = nums[queue.peek()];
            }
        }
        return res;
    }

    //    105. 从前序与中序遍历序列构造二叉树
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return buildTreeHelper(preorder, 0, preorder.length, inorder, 0, inorder.length);
    }

    private TreeNode buildTreeHelper(int[] preorder, int p_start, int p_end, int[] inorder, int i_start, int i_end) {
        // preorder 为空，直接返回 null
        if (p_start == p_end) {
            return null;
        }
        int root_val = preorder[p_start];
        TreeNode root = new TreeNode(root_val);
        //在中序遍历中找到根节点的位置
        int i_root_index = 0;
        for (int i = i_start; i < i_end; i++) {
            if (root_val == inorder[i]) {
                i_root_index = i;
                break;
            }
        }
        int leftNum = i_root_index - i_start;
        //递归的构造左子树
        root.left = buildTreeHelper(preorder, p_start + 1, p_start + leftNum + 1, inorder, i_start, i_root_index);
        //递归的构造右子树
        root.right = buildTreeHelper(preorder, p_start + leftNum + 1, p_end, inorder, i_root_index + 1, i_end);
        return root;
    }

    //    76. 最小覆盖子串
    public String minWindow(String s, String t) {
        if (s == null || t == null)
            return "";

        int i = 0, j = 1;
        String res = "";
        while (j < s.length()) {
            while (!subContains(s.substring(i, j), t) && j < s.length()) {
                j++;
            }
            while (subContains(s.substring(i, j), t) && i <= j) {
                String temp = s.substring(i, j);
                if (res == "") {
                    res = temp;
                } else {
                    if (res.length() > temp.length())
                        res = temp;
                }
                i++;
            }
        }
        return res;


    }


    Boolean subContains(String s, String t) {
        for (int i = 0; i < t.length(); i++) {
            String tem = t.substring(i, i + 1);
            if (s.contains(tem)) {
                s.replaceFirst(tem, "");
            } else {
                return false;
            }
        }
        return true;
    }

    //    129. 求根节点到叶节点数字之和
    public int sumNumbers(TreeNode root) {
        if (root == null)
            return -1;

        int res = 0;
        return dfs126(root, res);
    }

    public int dfs126(TreeNode root, int res) {
        if (root == null)
            return 0;
        res = res * 10 + root.val;
        int left = 0;
        int right = 0;
        if (root.left == null && root.right == null)
            return res;
        else {
            return dfs126(root.left, res) + dfs126(root.right, res);
        }
    }


    //    110. 平衡二叉树
    public boolean isBalanced(TreeNode root) {

        if (root == null)
            return true;

        return Math.abs(height(root.left) - height(root.right)) <= 1
                && isBalanced(root.left) && isBalanced(root.right);


    }

    public int height(TreeNode root) {
        if (root == null)
            return 0;
        else {
            return Math.max(height(root.left), height(root.left)) + 1;
        }


    }

    //    104. 二叉树的最大深度
    public int maxDepth(TreeNode root) {
        return dfs104(root);

    }

    public int dfs104(TreeNode root) {

        if (root == null)
            return 0;
        int right = 0, left = 0;
        if (root.right != null)
            right = dfs104(root.right);
        if (root.left != null)
            left = dfs104(root.left);

        return Math.max(right, left) + 1;

    }


    //    113. 路径总和 II
    List<List<Integer>> res1;

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {

        res1 = new ArrayList<>();
        LinkedList<Integer> state = new LinkedList<>();
        int sum = 0;
        dfs113(root, state, sum, targetSum);

        return res1;
    }

    public void dfs113(TreeNode treeNode, LinkedList<Integer> state, int sum, int targetSum) {
        if (treeNode == null)
            return;
        sum = sum + treeNode.val;
        state.add(treeNode.val);
        if (sum > targetSum)
            return;
        if (sum == targetSum && treeNode.left == null && treeNode.right == null) {
            res1.add(state);
            return;
        }
        if (treeNode.left != null)
            dfs113(treeNode.left, state, sum, targetSum);
        if (treeNode.right != null)
            dfs113(treeNode.right, state, sum, targetSum);
        state.removeLast();

    }


    //书的最大深度
    public int gexMaxDepth(TreeNode root) {
        if (root == null)
            return 0;
        int res = 0;
        int left = gexMaxDepth(root.left);
        int right = gexMaxDepth(root.right);
        res = Math.max(left, right) + 1;
        return res;
    }

    //543. 二叉树的直径
    int max1 = 0;

    public int diameterOfBinaryTree(TreeNode root) {

        return dfs543(root);
    }

    public int dfs543(TreeNode treeNode) {

        if (treeNode == null)
            return 0;
        int right = gexMaxDepth(treeNode.right);
        int left = gexMaxDepth(treeNode.left);
        int res = right + left + 1;
        max = Math.max(res, max);

        dfs543(treeNode.right);
        dfs543(treeNode.left);

        return max;

    }


    //    101. 对称二叉树
    public boolean isSymmetric(TreeNode root) {

        return dfs101(root, root);
    }

    public boolean dfs101(TreeNode treeNode1, TreeNode treeNode2) {

        if (treeNode1 == null && treeNode2 == null)
            return true;
        if (treeNode1 == null || treeNode2 == null)
            return false;

        if (treeNode1.val != treeNode2.val)
            return false;

        return dfs101(treeNode1.left, treeNode2.right) && dfs101(treeNode1.right, treeNode2.left);

    }

    //32. 最长有效括号
    //输入：s = ")()())"
    //输出：4
    //解释：最长有效括号子串是 "()()"
    public int longestValidParentheses(String s) {
        int maxans = 0;
        int temp = 0;
        Deque<Character> stack = new LinkedList<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push('(');
            } else {
                if (!stack.isEmpty()) {
                    stack.pop();
                    temp += 2;
                    maxans = Math.max(maxans, temp);
                } else {
                    temp = 0;
                }

            }
        }
        return maxans - stack.size() * 2;
    }

    public int longestValidParentheses1(String s) {
        int maxans = 0;
        Deque<Integer> stack = new LinkedList<Integer>();
        stack.push(-1);
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                stack.pop();
                if (stack.isEmpty()) {
                    stack.push(i);
                } else {
                    maxans = Math.max(maxans, i - stack.peek());
                }
            }
        }
        return maxans;
    }

    // 98. 验证二叉搜索树
    public boolean isValidBST(TreeNode root) {

        return dfs98(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    public boolean dfs98(TreeNode node, long lower, long upper) {
        if (node == null)
            return true;
        if (node.val <= lower || node.val >= upper)
            return false;

        return dfs98(node.left, lower, node.val) && dfs98(node.right, node.val, upper);

    }

    long pre = Long.MIN_VALUE;

    public boolean isValidBST1(TreeNode root) {
        if (root == null)
            return true;

        if (!isValidBST1(root.left)) {
            return false;
        }

        if (root.val >= pre) {
            return false;
        }
        pre = root.val;

        return isValidBST1(root.right);
    }


    @Test
    public void test12() {
        String a = ")()())";
        int res = longestValidParentheses(a);
        System.out.println(res);

    }


    // 零钱兑换问题
    public int coinChange(int[] coins, int amount) {
        int max = amount + 1;
        int[] dp = new int[max];
        Arrays.fill(dp, max);

        dp[0] = 0;

        for (int i = 1; i < max; i++) {
            for (int j = 0; j < coins.length; j++) {

                if (coins[j] <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }


    //    64. 最小路径和
    public int minPathSum(int[][] grid) {
        if (grid == null)
            return -1;

        int i = grid.length;
        int j = grid[0].length;

        int[][] dfs = new int[i][j];
        dfs[0][0] = grid[0][0];
        for (int m = 0; m < i; m++) {
            dfs[m][0] = dfs[m - 1][0] + grid[m][0];
        }
        for (int m = 0; m < j; m++) {
            dfs[0][m] = dfs[0][m - 1] + grid[0][m];
        }
        for (int m = 0; m < i; m++) {
            for (int n = 0; n < j; n++) {
                dfs[m][n] = Math.min(dfs[m - 1][n], dfs[m][n - 1]) + grid[m][n];
            }
        }
        return dfs[i - 1][j - 1];
    }

    //76 最小覆盖字符串
    public String minWindow1(String s, String t) {
        if (s == null || t == null)
            return null;
        String res = s;
        int i = 0, j = 0;
        while (j < s.length()) {
            while (!subContains1(s.substring(i, j), t) && j < s.length()) {
                j++;
            }
            while (subContains1(s.substring(i, j), t) && j < j) {
                if (s.substring(i, j).length() < res.length()) {
                    res = s.substring(i, j);
                }
                i++;
            }
        }
        return res;

    }


    Boolean subContains1(String s, String t) {
        for (int i = 0; i < t.length(); i++) {
            String tem = t.substring(i, i + 1);
            if (s.contains(tem)) {
                s = s.replaceFirst(tem, "");
                System.out.println(s);
            } else {
                return false;
            }
        }
        return true;
    }

    @Test
    public void test121() {
        String a = ")()())";
        subContains1("aaaa", "aa");
        System.out.println(a);
    }

    //    34. 在排序数组中查找元素的第一个和最后一个位置
    public int[] searchRange(int[] nums, int target) {
        int[] res = new int[]{-1, -1};
        if (nums == null)
            return res;

        int i = 0, j = nums.length;
        int index = -1;

        while (i < j) {
            int mid = i + (j - i) / 2;
            if (nums[mid] == target) {
                index = mid;
                break;
            } else if (nums[mid] > target) {
                j = mid - 1;
            } else {
                i = mid + 1;
            }
        }
        if (index != -1) {
            int a = index;
            while (nums[a] == target && a >= 0) {
                a--;
            }
            int b = index;
            while (nums[b] == target && b < nums.length) {
                b++;
            }
            res[0] = a;
            res[1] = b;
        }

        return res;


    }

    //    226. 翻转二叉树、
    public TreeNode invertTree(TreeNode root) {

        if (root == null)
            return null;
        TreeNode head = root;
        dfs226(head);
        return root;

    }

    public void dfs226(TreeNode root) {
        if (root == null)
            return;
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        dfs226(root.left);
        dfs226(root.right);
    }


    //    718. 最长重复子数组
    int[][] memo718;

    public int findLength(int[] nums1, int[] nums2) {
        if (nums1 == null || nums2 == null)
            return -1;
        int i = nums1.length;
        int j = nums2.length;

        memo718 = new int[i][j];

        return dfs718(nums1, nums2, i - 1, j - 1);

    }

    public int dfs718(int[] nums1, int[] nums2, int i, int j) {
        if (i < 0 || j < 0) {
            return 0;
        }
        if (memo718[i][j] != 0)
            return memo718[i][j];
        int res = 0;
        if (nums1[i] == nums2[j]) {
            res = dfs718(nums1, nums2, i - 1, j - 1) + 1;
        } else {
            res = Math.max(dfs718(nums1, nums2, i - 1, j), dfs718(nums1, nums2, i, j - 1));
        }
        memo718[i][j] = res;
        return res;
    }


    //    112. 路径总和 II
    boolean res111;

    public boolean hasPathSum(TreeNode root, int targetSum) {

        res111 = false;
        LinkedList<Integer> state = new LinkedList<>();
        int sum = 0;
        dfs111(root, state, sum, targetSum);

        return res111;
    }

    public void dfs111(TreeNode treeNode, LinkedList<Integer> state, int sum, int targetSum) {
        if (res111 == true)
            return;
        if (treeNode == null)
            return;
        sum = sum + treeNode.val;
//        state.add(treeNode.val);
        if (sum > targetSum)
            return;
        if (sum == targetSum && treeNode.left == null && treeNode.right == null) {
//            res1.add(state);
            res111 = true;
            return;
        }
        if (treeNode.left != null)
            dfs113(treeNode.left, state, sum, targetSum);
        if (treeNode.right != null)
            dfs113(treeNode.right, state, sum, targetSum);
//        state.removeLast();
    }

    public void swap(int nums[], int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public int partion(int[] nums, int left, int right) {
        int tempIndex = right;
        int value = nums[right];

        while (left < right) {
            while (nums[left] < value && left < right) {
                left++;
            }
            while (nums[right] >= value && left < right) {
                right--;
            }

            swap(nums, left, right);

        }
        swap(nums, tempIndex, left);
        return left;
    }

    public void quickSort(int[] nums, int left, int right) {

        if (left >= right)
            return;

        int mid = partion(nums, left, right);
        quickSort(nums, left, mid - 1);
        quickSort(nums, mid + 1, right);

    }

    @Test
    public void ptest() {
        int nums[] = new int[]{3, 4, 2, 5, 7, 4, 8, 8, 9, 34, 5, 646, 3, 996};
        quickSort(nums, 0, nums.length - 1);
        System.out.println(Arrays.toString(nums));
    }

    //    169. 多数元素
    public int majorityElement(int[] nums) {
        if (nums == null)
            return 0;
        quickSort(nums, 0, nums.length - 1);
        return nums[nums.length / 2];
    }


    List<List<Integer>> ress = new ArrayList<>();

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        if (candidates == null) return ress;
        dfs(target, 0, new Stack<Integer>(), candidates);
        return ress;
    }

    //深度遍历
    private void dfs(int target, int index, Stack<Integer> pre, int[] candidates) {
        //等于零说明结果符合要求
        if (target == 0) {
            ress.add(new ArrayList<>(pre));
            return;
        }
        //遍历，index为本分支上一节点的减数的下标
        for (int i = index; i < candidates.length; i++) {
            //如果减数大于目标值，则差为负数，不符合结果
            if (candidates[i] <= target) {
                pre.push(candidates[i]);
                //目标值减去元素值
                dfs(target - candidates[i], i, pre, candidates);
                //每次回溯将最后一次加入的元素删除
                pre.pop();
            }
        }
    }

    @Test
    public void ptest1() {
        int nums[] = new int[]{3, 4, 2};

        List<List<Integer>> combinationSum1 = combinationSum(nums, 10);

        System.out.println(combinationSum1);
    }


    //    221. 最大正方形
    public int maximalSquare(char[][] matrix) {
        int maxRes = 0;
        if (matrix == null)
            return maxRes;
        int columns = matrix.length;
        int rows = matrix[0].length;
        int[][] dp = new int[columns][rows];

        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i - 1][j - 1]), dp[i][j - 1]) + 1;
                    }
                    maxRes = Math.max(maxRes, dp[i][j]);
                }
            }
        }

        return maxRes * maxRes;
    }

    //    83. 删除排序链表中的重复元素
    public ListNode deleteDuplicates1(ListNode head) {
        if (head == null)
            return null;
        ListNode newHead = new ListNode(Integer.MAX_VALUE, head);

        ListNode cur = newHead.next, pre = newHead;
        while (cur != null) {
            if (cur.val == pre.val) {
                ListNode temp = cur.next;
                pre.next = temp;
                cur = temp;
            } else {
                cur = cur.next;
                pre = pre.next;
            }
        }
        return newHead.next;
    }

    //62. 不同路径
    public int uniquePaths(int m, int n) {
        if (m < 1 || n < 1)
            return 0;

        int[][] dp = new int[m][n];

        for (int i = 0; i < m; i++)
            dp[i][0] = 1;
        for (int i = 0; i < n; i++)
            dp[0][i] = 1;
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }


    //    128 最长连续序列
    public int longestConsecutive(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        if (nums.length == 1)
            return 1;

        HashMap<Integer, Integer> hashMap = new HashMap<>();

        Arrays.sort(nums);

        int res = 1;
        int cur = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == nums[i - 1] + 1) {
                cur += 1;
                res = Math.max(res, cur);
            } else if (nums[i] == nums[i - 1]) {
            } else {
                cur = 1;
            }
        }
        return res;
    }


    //240. 搜索二维矩阵 II
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null)
            return false;

        int rows = matrix.length - 1;
        int columns = matrix[0].length - 1;
        int column = columns, row = 0;

        while (column >= 0 && row <= rows) {

            if (matrix[row][column] == target) {
                return true;
            } else if (matrix[row][column] > target) {
                column--;
            } else {
                row++;
            }
        }
        return false;
    }

}


