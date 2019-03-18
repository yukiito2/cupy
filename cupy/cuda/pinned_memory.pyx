# distutils: language = c++

import collections
import ctypes
import gc
import warnings
import weakref

from fastrlock cimport rlock
from libcpp cimport bool
from libcpp cimport algorithm

from cupy.cuda import runtime

from cupy.core cimport internal
from cupy.cuda cimport runtime


cdef class PinnedChunk:

    def __init__(self, mem, Py_ssize_t offset, Py_ssize_t size):
        assert mem.ptr > 0 or offset == 0
        self.mem = mem
        self.ptr = mem.ptr + offset
        self.offset = offset
        self.size = size
        self.prev = None
        self.next = None


cdef class PinnedMemory:

    """Pinned memory allocation on host.

    This class provides a RAII interface of the pinned memory allocation.

    Args:
        size (int): Size of the memory allocation in bytes.

    """

    def __init__(self, Py_ssize_t size, unsigned int flags=0):
        self.size = size
        self.ptr = 0
        if size > 0:
            self.ptr = runtime.hostAlloc(size, flags)

    def __dealloc__(self):
        if self.ptr:
            runtime.freeHost(self.ptr)

    def __int__(self):
        """Returns the pointer value to the head of the allocation."""
        return self.ptr


cdef class PinnedMemoryPointer:

    """Pointer of a pinned memory.

    An instance of this class holds a reference to the original memory buffer
    and a pointer to a place within this buffer.

    Args:
        mem (PinnedMemory): The device memory buffer.
        offset (int): An offset from the head of the buffer to the place this
            pointer refers.

    Attributes:
        mem (PinnedMemory): The device memory buffer.
        ptr (int): Pointer to the place within the buffer.
    """

    def __init__(self, PinnedMemory mem, Py_ssize_t offset):
        self.mem = mem
        self.ptr = mem.ptr + offset

    def __int__(self):
        """Returns the pointer value."""
        return self.ptr

    def __add__(x, y):
        """Adds an offset to the pointer."""
        cdef PinnedMemoryPointer self
        cdef Py_ssize_t offset
        if isinstance(x, PinnedMemoryPointer):
            self = x
            offset = <Py_ssize_t?>y
        else:
            self = <PinnedMemoryPointer?>y
            offset = <Py_ssize_t?>x
        return PinnedMemoryPointer(
            self.mem, self.ptr - self.mem.ptr + offset)

    def __iadd__(self, Py_ssize_t offset):
        """Adds an offset to the pointer in place."""
        self.ptr += offset
        return self

    def __sub__(self, offset):
        """Subtracts an offset from the pointer."""
        return self + -offset

    def __isub__(self, Py_ssize_t offset):
        """Subtracts an offset from the pointer in place."""
        return self.__iadd__(-offset)

    cpdef copy_from_device(self, memory.MemoryPointer src, Py_ssize_t size):
        """Copies data from src (device memory) to self (pinned memory).
        Copied from anaruse's repository
        Source: https://github.com/anaruse/cupy/blob/OOC_cupy_v102/cupy/cuda/pinned_memory.pyx

        Args:
            src (cupy.cuda.MemoryPointer): Source memory pointer.
            size (int): Size of data in bytes.
        """
        if size > 0:
            runtime.memcpy(self.ptr, src.ptr, size,
                           runtime.memcpyDeviceToHost)

    cpdef copy_from_device_async(self, memory.MemoryPointer src,
                                 Py_ssize_t size, stream):
        """Copies data from src (device memory) to self (pinned memory)
        asynchronously.
        Copied from anaruse's repository
        Source: https://github.com/anaruse/cupy/blob/OOC_cupy_v102/cupy/cuda/pinned_memory.pyx

        Args:
            src (cupy.cuda.MemoryPointer): Source memory pointer.
            size (int): Size of data in bytes.
            stream (cupy.cuda.Stream): CUDA stream.
        """
        if size > 0:
            runtime.memcpyAsync(self.ptr, src.ptr, size,
                                runtime.memcpyDeviceToHost, stream.ptr)

    cpdef copy_to_device(self, memory.MemoryPointer dst, Py_ssize_t size):
        """Copies data from self (pinned memory) to dst (device memory).
        Copied from anaruse's repository
        Source: https://github.com/anaruse/cupy/blob/OOC_cupy_v102/cupy/cuda/pinned_memory.pyx

        Args:
            dst (cupy.cuda.MemoryPointer): Destination memory pointer.
            size (int): Size of data in bytes.
        """
        if size > 0:
            runtime.memcpy(dst.ptr, self.ptr, size,
                           runtime.memcpyHostToDevice)

    cpdef copy_to_device_async(self, memory.MemoryPointer dst,
                               Py_ssize_t size, stream):
        """Copies data from self (pinned memory) to dst (device memory)
        asynchronously.
        Copied from anaruse's repository
        Source: https://github.com/anaruse/cupy/blob/OOC_cupy_v102/cupy/cuda/pinned_memory.pyx

        Args:
            dst (cupy.cuda.MemoryPointer): Destination memory pointer.
            size (int): Size of data in bytes.
        """
        if size > 0:
            runtime.memcpyAsync(dst.ptr, self.ptr, size,
                                runtime.memcpyHostToDevice, stream.ptr)

    cpdef Py_ssize_t size(self):
        return self.mem.size - (self.ptr - self.mem.ptr)
    def __getbuffer__(self, Py_buffer *buffer, int flags):
        size = self.size()

        self._shape[0] = size
        self._strides[0] = 1

        buffer.buf = <void*>self.ptr
        buffer.format = 'b'
        buffer.internal = NULL
        buffer.itemsize = 1
        buffer.len = size
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self._shape
        buffer.strides = self._strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __getsegcount__(self, Py_ssize_t *lenp):
        if lenp != NULL:
            lenp[0] = self.size()
        return 1

    def __getreadbuffer__(self, Py_ssize_t idx, void **p):
        if idx != 0:
            raise SystemError("accessing non-existent buffer segment")
        p[0] = <void*>self.ptr
        return self.size()

    def __getwritebuffer__(self, Py_ssize_t idx, void **p):
        if idx != 0:
            raise SystemError("accessing non-existent buffer segment")
        p[0] = <void*>self.ptr
        return self.size()


cdef class _EventWatcher:
    cdef:
        cdef list events
        cdef object _lock

    def __init__(self):
        self.events = []
        self._lock = rlock.create_fastrlock()

    cpdef add(self, event, obj):
        """ Add event to be monitored.
        The ``obj`` are automatically released when the event done.
        Args:
            event (cupy.cuda.Event): The CUDA event to be monitored.
            obj: The object to be held.
        """
        rlock.lock_fastrlock(self._lock, -1, True)
        try:
            self._check_and_release_without_lock()
            if event.done:
                return
            self.events.append((event, obj))
        finally:
            rlock.unlock_fastrlock(self._lock)

    cpdef check_and_release(self):
        """ Check and release completed events.
        """
        if not self.events:
            return
        rlock.lock_fastrlock(self._lock, -1, True)
        try:
            self._check_and_release_without_lock()
        finally:
            rlock.unlock_fastrlock(self._lock)

    cpdef _check_and_release_without_lock(self):
        while self.events and self.events[0][0].done:
            del self.events[0]


cpdef PinnedMemoryPointer _mallochost(Py_ssize_t size):
    mem = PinnedMemory(size, runtime.hostAllocPortable)
    return PinnedMemoryPointer(mem, 0)


cdef object _current_allocator = _mallochost
cdef _EventWatcher _watcher = _EventWatcher()


cpdef _add_to_watch_list(event, obj):
    """ Add event to be monitored.
    The ``obj`` are automatically released when the event done.
    Args:
        event (cupy.cuda.Event): The CUDA event to be monitored.
        obj: The object to be held.
    """
    _watcher.add(event, obj)


cpdef PinnedMemoryPointer alloc_pinned_memory(Py_ssize_t size):
    """Calls the current allocator.
    Use :func:`~cupy.cuda.set_pinned_memory_allocator` to change the current
    allocator.
    Args:
        size (int): Size of the memory allocation.
    Returns:
        ~cupy.cuda.PinnedMemoryPointer: Pointer to the allocated buffer.
    """
    _watcher.check_and_release()
    return _current_allocator(size)


cpdef set_pinned_memory_allocator(allocator=_mallochost):
    """Sets the current allocator for the pinned memory.
    Args:
        allocator (function): CuPy pinned memory allocator. It must have the
            same interface as the :func:`cupy.cuda.alloc_pinned_memory`
            function, which takes the buffer size as an argument and returns
            the device buffer of that size. When ``None`` is specified, raw
            memory allocator is used (i.e., memory pool is disabled).
    """
    global _current_allocator
    _current_allocator = allocator


class PinnedPooledMemory(PinnedMemory):

    """Memory allocation for a memory pool.
    As the instance of this class is created by memory pool allocator, users
    should not instantiate it manually.
    """

    def __init__(self, PinnedChunk chunk, pool):
        self.ptr = chunk.ptr
        self.size = chunk.size
        self.pool = pool

    def __del__(self):
        if self.ptr != 0:
            self.free()

    def free(self):
        """Releases the memory buffer and sends it to the memory pool.
        This function actually does not free the buffer. It just returns the
        buffer to the memory pool for reuse.
        """
        pool = self.pool()
        if pool and self.ptr != 0:
            pool.free(self.ptr, self.size)
        self.ptr = 0
        self.size = 0


cdef class PinnedMemoryPool:

    """Memory pool for pinned memory on the host.
    Note that it preserves all allocated memory buffers even if the user
    explicitly release the one. Those released memory buffers are held by the
    memory pool as *free blocks*, and reused for further memory allocations of
    the same size.
    Args:
        allocator (function): The base CuPy pinned memory allocator. It is
            used for allocating new blocks when the blocks of the required
            size are all in use.
    """

    def __init__(self, allocator=_mallochost):
        self._allocation_unit_size = 512
        self._in_use = {}
        self._in_use_memptr = {}
        self._free = []
        self._allocator = allocator
        self._weakref = weakref.ref(self)
        self._free_lock = rlock.create_fastrlock()
        self._in_use_lock = rlock.create_fastrlock()

        _, total_device_memory = runtime.memGetInfo()
        self.malloc(int(total_device_memory*5))

    cpdef Py_ssize_t _round_size(self, Py_ssize_t size):
        """Round up the memory size to fit memory alignment of cudaMalloc."""
        unit = self._allocation_unit_size
        return ((size + unit - 1) // unit) * unit

    cpdef int _bin_index_from_size(self, Py_ssize_t size):
        """Get appropriate bins index from the memory size"""
        unit = self._allocation_unit_size
        return (size - 1) // unit

    cpdef _append_to_free_list(self, Py_ssize_t size, chunk):
        cdef int index, bin_index
        cdef set free_list
        bin_index = self._bin_index_from_size(size)
        
        rlock.lock_fastrlock(self._free_lock, -2, True)
        try:
            index = algorithm.lower_bound(
                self._index.begin(), self._index.end(),
                bin_index) - self._index.begin()
            if index < self._index.size() and self._index[index] == bin_index:
                free_list = self._free[index]
            else:
                free_list = set()
                self._index.insert(
                    self._index.begin() + index, bin_index)
                self._free.insert(index, free_list)
            free_list.add(chunk)
        finally:
            rlock.unlock_fastrlock(self._free_lock)

    cpdef bint _remove_from_free_list(self, Py_ssize_t size, chunk) except *:
        cdef int index, bin_index
        cdef set free_list
        bin_index = self._bin_index_from_size(size)
        rlock.lock_fastrlock(self._free_lock, -2, True)
        try:
            index = algorithm.lower_bound(
                self._index.begin(), self._index.end(),
                bin_index) - self._index.begin()
            if self._index[index] != bin_index:
                return False
            free_list = self._free[index]
            if chunk in free_list:
                free_list.remove(chunk)
                return True
        finally:
            rlock.unlock_fastrlock(self._free_lock)
        return False

    cpdef tuple _split(self, PinnedChunk chunk, Py_ssize_t size):
        """Split contiguous block of a larger allocation"""
        cdef PinnedChunk head
        cdef PinnedChunk remaining

        assert chunk.size >= size
        if chunk.size == size:
            return chunk, None

        head = PinnedChunk(chunk.mem, chunk.offset, size)
        remaining = PinnedChunk(chunk.mem, chunk.offset + size, chunk.size - size)
        if chunk.prev is not None:
            head.prev = chunk.prev
            chunk.prev.next = head
        if chunk.next is not None:
            remaining.next = chunk.next
            chunk.next.prev = remaining
        head.next = remaining
        remaining.prev = head
        return head, remaining

    cpdef PinnedChunk _merge(self, PinnedChunk head, PinnedChunk remaining):
        """Merge previously splitted block (chunk)"""
        cdef PinnedChunk merged
        size = head.size + remaining.size
        merged = PinnedChunk(head.mem, head.offset, size)
        if head.prev is not None:
            merged.prev = head.prev
            merged.prev.next = merged
        if remaining.next is not None:
            merged.next = remaining.next
            merged.next.prev = merged
        return merged

    cpdef PinnedMemoryPointer _alloc(self, Py_ssize_t rounded_size):

        return self._allocator(rounded_size)

    cpdef PinnedMemoryPointer malloc(self, Py_ssize_t size):
        rounded_size = self._round_size(size)
        return self._malloc(rounded_size)

    cpdef PinnedMemoryPointer _malloc(self, Py_ssize_t size):
        
        cdef set free_list
        cdef PinnedChunk chunk = None
        cdef PinnedChunk remaining = None
        cdef int bin_index, index, length
        
        cdef Py_ssize_t chunk_list_free_size = 0
        
        if size == 0:
            return PinnedMemoryPointer(PinnedMemory(0), 0)

        # find best-fit, or a smallest larger allocation
        rlock.lock_fastrlock(self._free_lock, -2, True)
        bin_index = self._bin_index_from_size(size)
        try:
            index = algorithm.lower_bound(
                self._index.begin(), self._index.end(),
                bin_index) - self._index.begin()
            length = self._index.size()
            for i in range(index, length):
                free_list = self._free[i]
                if free_list:
                    chunk = free_list.pop()
                    break
        finally:
            rlock.unlock_fastrlock(self._free_lock)
        
        if chunk is not None:
            chunk, remaining = self._split(chunk, size)
        else:
            # cudaMallocHost if a cache is not found
            try:
                mem = self._alloc(size).mem
                chunk = PinnedChunk(mem, 0, size)
            except runtime.CUDARuntimeError as e:
                runtime.deviceSynchronize()
                if e.status != runtime.errorMemoryAllocation:
                    raise

        rlock.lock_fastrlock(self._in_use_lock, -2, True)
        try:
            self._in_use[chunk.ptr] = chunk
        finally:
            rlock.unlock_fastrlock(self._in_use_lock)
        if remaining is not None:
            self._append_to_free_list(remaining.size, remaining)
        pmem = PinnedPooledMemory(chunk, self._weakref)
        memptr = PinnedMemoryPointer(pmem, 0)
        self._in_use_memptr[chunk.ptr] = weakref.ref(memptr)
        
        return memptr

    cpdef free(self, size_t ptr, Py_ssize_t size):
        cdef set free_list
        cdef PinnedChunk chunk

        #print("free: {}".format(size))

        rlock.lock_fastrlock(self._in_use_lock, -2, True)
        try:
            chunk = self._in_use.pop(ptr, None)
        finally:
            rlock.unlock_fastrlock(self._in_use_lock)
        if chunk is None:
            raise RuntimeError('Cannot free out-of-pool memory')

        if chunk.next is not None:
            if self._remove_from_free_list(chunk.next.size, chunk.next):
                chunk = self._merge(chunk, chunk.next)

        if chunk.prev is not None:
            if self._remove_from_free_list(chunk.prev.size, chunk.prev):
                chunk = self._merge(chunk.prev, chunk)

        self._append_to_free_list(chunk.size, chunk)


    cpdef free_all_blocks(self):
        cdef set free_list, keep_list
        cdef PinnedChunk chunk
        
        # Free all **non-split** chunks
        rlock.lock_fastrlock(self._free_lock, -2, True)
        try:
            for i in range(len(self._free)):
                free_list = self._free[i]
                keep_list = set()
                for chunk in free_list:
                    if chunk.prev is not None or chunk.next is not None:
                        keep_list.add(chunk)
                self._free[i] = keep_list
        finally:
            rlock.unlock_fastrlock(self._free_lock)

    cpdef free_all_free(self):
        warnings.warn(
            'free_all_free is deprecated. Use free_all_blocks instead.',
            DeprecationWarning)
        self.free_all_blocks()

    cpdef n_free_blocks(self):
        cdef Py_ssize_t n = 0
        cdef set free_list
        rlock.lock_fastrlock(self._free_lock, -2, True)
        try:
            for free_list in self._free:
                n += len(free_list)
        finally:
            rlock.unlock_fastrlock(self._free_lock)
        return n