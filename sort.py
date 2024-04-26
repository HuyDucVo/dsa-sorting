import bisect
import random
import time
import sys
import multiprocessing

sys.setrecursionlimit(100000)


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result

    return timed


@timeit
def generate_random_array():
    return [random.randint(1, 100_000_000) for _ in range(100_000)]


@timeit
def bubbleSort(arr):
    N = len(arr)
    for _ in range(N):
        for i in range(1, N):
            if arr[i] < arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
    return arr


@timeit
def insertionSort(arr):
    for i in range(1, len(arr)):
        j = i
        while j > 0 and arr[j - 1] > arr[j]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j = j - 1
    return arr


@timeit
def selectionSort(arr):
    N = len(arr)
    for i in range(N):
        min_index = i
        for j in range(i + 1, N):
            if arr[min_index] > arr[j]:
                min_index = j

        if min_index != i:
            arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr


@timeit
def mergeSort(arr):
    def mergeSortRunner(arr):
        def merge(left, right):
            ret = []
            i, j = 0, 0
            N = len(left)
            M = len(right)
            while i < N and j < M:
                if left[i] < right[j]:
                    ret.append(left[i])
                    i += 1
                else:
                    ret.append(right[j])
                    j += 1

            theRest = left[i:] if i < N else right[j:]
            ret.extend(theRest)
            return ret

        N = len(arr)
        if N <= 1:
            return arr

        arrCheck = mergeSortRunner(arr[:N // 2])
        arr2 = mergeSortRunner(arr[N // 2:])
        return merge(arrCheck, arr2)

    return mergeSortRunner(arr)


@timeit
def quickSort(arr):
    def partition(par, left, right):
        pivot = par[right]
        i = left

        for j in range(left, right):
            if par[j] <= pivot:
                par[i], par[j] = par[j], par[i]
                i += 1
        par[right], par[i] = par[i], par[right]
        return i

    def quickSort(par, left, right):
        if left < right:
            pivot = partition(par, left, right)
            quickSort(par, left, pivot - 1)
            quickSort(par, pivot + 1, right)

    quickSort(arr, 0, len(arr) - 1)
    return arr


@timeit
def quicksortIterative(arr):
    pass


@timeit
def heapsort(arr):
    def heapify(arr, bound, i):
        left = 2 * i + 1
        right = 2 * i + 2
        max = i
        if left < bound and arr[i] < arr[left]:
            max = left

        if right < bound and arr[max] < arr[right]:
            max = right

        if max != i:
            arr[i], arr[max] = arr[max], arr[i]
            heapify(arr, bound, max)

    def buildHeap(arr):
        n = len(arr)
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)

    N = len(arr)
    buildHeap(arr)
    for boundRight in range(N - 1, 0, -1):
        arr[boundRight], arr[0] = arr[0], arr[boundRight]
        heapify(arr, boundRight, 0)
    return arr


@timeit
def timSort(arr):
    MIN_MERGE = 32

    def insertionSort(arr, start, end):
        for i in range(start, end):
            j = i
            while j > start and arr[j] < arr[j - 1]:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]
                j -= 1

    def calculateMinRun(n):
        r = 0
        while n >= MIN_MERGE:
            r |= n & 1
            n >>= 1
        return n + r

    def merge(arr, l, m, r):
        left, right = arr[l:m].copy(), arr[m:r].copy()
        i, j, k = 0, 0, l

        galloping = False
        gallopingLeft = 0
        gallopingRight = 0

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                if galloping:
                    gallopingLeft |= 1 << i
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            arr[k] = left[i]
            k += 1
            i += 1

        while j < len(right):
            arr[k] = right[j]
            k += 1
            j += 1

    N = len(arr)
    minRun = calculateMinRun(N)
    for start in range(0, N, minRun):
        end = min(start + minRun, N)
        insertionSort(arr, start, end)

    size = minRun
    while size < N:
        for left in range(0, N, 2 * size):
            mid = min(N, left + size)
            right = min(left + 2 * size, N)
            if mid < right:
                merge(arr, left, mid, right)

        size = 2 * size

    return arr


@timeit
def treeSort(arr):
    class Node:
        def __init__(self, item=0):
            self.item = item
            self.left = None
            self.right = None

    root = Node(arr[0])

    def insert(val):
        def insertHelper(root):
            if not root:
                return Node(val)

            if val < root.item:
                root.left = insertHelper(root.left)
            else:
                root.right = insertHelper(root.right)
            return root

        nonlocal root
        return insertHelper(root)

    ans = []

    def inOrder(root):
        if not root:
            return
        inOrder(root.left)
        ans.append(root.item)
        inOrder(root.right)

    for val in arr[1:]:
        insert(val)

    inOrder(root)
    return ans


@timeit
def shellsort(arr):
    size = len(arr)
    gap = size // 2
    while gap > 0:
        j = gap
        while j < size:
            i = j - gap
            while i >= 0:
                if arr[i + gap] < arr[i]:
                    arr[i], arr[i + gap] = arr[i + gap], arr[i]
                else:
                    break
                i = i - gap
            j += 1
        gap = gap // 2

    return arr


@timeit
def bucketSort(arr):
    maxVal = max(arr)
    length = len(arr)
    coeff = maxVal // length
    buckets = [[] for _ in range(length)]
    for val in arr:
        indexBucket = (val // coeff)
        if indexBucket >= length:
            indexBucket = length - 1

        if not buckets[indexBucket]:
            buckets[indexBucket].append(val)
        else:
            buckets[indexBucket].insert(bisect.bisect_left(buckets[indexBucket], val), val)

    return [val for bucket in buckets for val in bucket]


@timeit
def radixSort(arr):
    maxVal = max(arr)
    exp = 1
    while exp <= maxVal:
        radixArrs = [[] for _ in range(10)]
        while arr:
            radixArrs[(arr[-1] // exp) % 10].append(arr.pop())

        [arr.extend(radixArr[::-1]) for radixArr in radixArrs]

        exp *= 10

    return arr


@timeit
def countingSort(arr):
    maxVal = max(arr)
    minVal = min(arr)
    counts = [0] * (maxVal - minVal + 1)
    ans = [0] * len(arr)

    for num in arr:
        counts[num - minVal] += 1

    for i in range(1, len(counts)):
        counts[i] += counts[i - 1]

    for i in range(len(arr) - 1, -1, -1):
        ans[counts[arr[i] - minVal] - 1] = arr[i]
        counts[arr[i] - minVal] -= 1

    return ans

def cubesort(arr):
    def merge(left, right):
        result = []
        l = r = 0

        while l < len(left) and r < len(right):
            if left[l] < right[r]:
                result.append(left[l])
                l += 1
            else:
                result.append(right[r])
                r += 1

            result.extend(left[l:])
            result.extend(right[r:])
            return result

    if len(arr) <= 1:
        return arr

    size = len(arr) // 3

    left = arr[:size]
    middle = arr[size:2 * size]
    right = arr[2 * size:]

    if len(left) < 5:
        left = sorted(left)

    if len(middle) < 5:
        middle = sorted(middle)

    if len(right) < 5:
        right = sorted(right)

    with multiprocessing.Pool(processes=3) as pool:
        left = pool.apply_async(cubesort, [left])
        middle = pool.apply_async(cubesort, [middle])
        right = pool.apply_async(cubesort, [right])

        left = left.get()
        middle = middle.get()
        right = right.get()

    return merge(merge(left, middle), right)


ARR = generate_random_array()
arrCheck = sorted(ARR[:])

allSort = [
    bubbleSort,
    insertionSort,
    selectionSort,
    mergeSort,
    quickSort,
    heapsort,
    timSort,
    treeSort,
    shellsort,
    bucketSort,
    radixSort,
    countingSort
]

for sortAlgorithm in allSort:
    assert (sortAlgorithm(ARR.copy()) == arrCheck)
