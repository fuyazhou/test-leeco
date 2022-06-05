import org.junit.Test;

import java.util.Arrays;

public class Sort {


    /*
    冒泡排序算法的运作如下：
    比较相邻的元素，如果前一个比后一个大，就把它们两个调换位置。
    对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对。这步做完后，最后的元素会是最大的数。
    针对所有的元素重复以上的步骤，除了最后一个。
    持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。*/
    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public void bubbleSort(int[] nums) {
        if (nums == null)
            return;
        for (int i = 0; i < nums.length; i++) {
            for (int j = 0; j < nums.length - 1 - i; j++) {
                if (nums[j] > nums[j + 1])
                    swap(nums, j, j + 1);
            }
        }
    }

    @Test
    public void bubbleSortTest() {
        int nums[] = new int[]{3, 4, 2, 5, 7, 4, 8, 8, 9, 34, 5, 646, 3, 463};
        bubbleSort(nums);
        System.out.println(Arrays.toString(nums));
    }


//    快速排序是由东尼·霍尔所发展的一种排序算法。在平均状况下，排序n个元素要O(nlogn)次比较。在最坏状况下则需要O(n^2)次比较，但这种状况并不常见。事实上，快速排序通常明显比其他O(nlogn)算法更快，因为它的内部循环可以在大部分的架构上很有效率地被实现出来。
//
//    快速排序使用分治策略(Divide and Conquer)来把一个序列分为两个子序列。步骤为：
//
//    从序列中挑出一个元素，作为"基准"(pivot).
//    把所有比基准值小的元素放在基准前面，所有比基准值大的元素放在基准的后面（相同的数可以到任一边），这个称为分区(partition)操作。
//    对每个分区递归地进行步骤1~2，递归的结束条件是序列的大小是0或1，这时整体已经被排好序了。

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

    public void quickSort(int[] nums, int left, int right) {
        if (left >= right)
            return;
        int index = partition(nums, left, right);

        quickSort(nums, left, index - 1);
        quickSort(nums, index + 1, right);
    }

    @Test
    public void ptest() {
        int nums[] = new int[]{3, 4, 2, 5, 7, 4, 8, 8, 9, 34, 5, 646, 3, 996};
        quickSort(nums, 0, nums.length - 1);
        System.out.println(Arrays.toString(nums));
    }

    //归并排序
    void Merge(int A[], int left, int mid, int right)// 合并两个已排好序的数组A[left...mid]和A[mid+1...right]
    {
        int[] temp = new int[left + right + 1];
        int index = 0;
        int i = left, j = mid + 1;
        while (i <= mid && j <= right) {
            if (A[i] >= A[j]) {
                temp[index] = A[j];
                j++;
            } else {
                temp[index] = A[i];
                i++;
            }
            index++;
        }
        while (i <= mid) {
            temp[index] = A[i];
            index++;
            i++;
        }
        while (j <= right) {
            temp[index] = A[j];
            index++;
            j++;
        }
        int len = right - left + 1;
        for (int m = 0; m < len; m++) {
            A[left] = temp[m];
            left++;
        }

    }

    public void meregeSort(int[] nums, int left, int right) {
        if (left >= right)
            return;

        int mid = (right + left) / 2;
        meregeSort(nums,left,mid);
        meregeSort(nums,mid+1,right);
        Merge(nums,left,mid,right);


    }


    @Test
    public void ptest1() {
        int nums[] = new int[]{3, 4, 2, 5, 7, 4, 8, 8, 9, 34, 5, 646, 3, 996};
        meregeSort(nums, 0, nums.length - 1);
        System.out.println(Arrays.toString(nums));
    }


}
