from cupy.cuda cimport memory
from libcpp cimport bool
from libcpp cimport vector

cdef class PinnedChunk:

    cdef:
        readonly object mem
        readonly size_t ptr
        readonly Py_ssize_t offset
        readonly Py_ssize_t size
        public PinnedChunk prev
        public PinnedChunk next
        public bint in_use
        object __weakref__


cdef class PinnedMemory:

    cdef:
        public size_t ptr
        public Py_ssize_t size


cdef class PinnedMemoryPointer:

    cdef:
        readonly object mem
        readonly size_t ptr
        Py_ssize_t _shape[1]
        Py_ssize_t _strides[1]
        object __weakref__

    cpdef Py_ssize_t size(self)
    cpdef copy_from_device(self, memory.MemoryPointer src, Py_ssize_t size)
    cpdef copy_from_device_async(self, memory.MemoryPointer src, Py_ssize_t size, stream)
    cpdef copy_to_device(self, memory.MemoryPointer dst, Py_ssize_t size)
    cpdef copy_to_device_async(self, memory.MemoryPointer dst, Py_ssize_t size, stream)


cpdef _add_to_watch_list(event, obj)


cpdef PinnedMemoryPointer alloc_pinned_memory(Py_ssize_t size)


cpdef set_pinned_memory_allocator(allocator=*)


cdef class PinnedMemoryPool:

    cdef:
        object _allocator
        dict _in_use
        dict _in_use_memptr
        list _free
        object __weakref__
        object _weakref
        object _free_lock
        object _in_use_lock
        readonly Py_ssize_t _allocation_unit_size
        readonly Py_ssize_t _initial_bins_size
        vector.vector[int] _index

    cpdef PinnedMemoryPointer _alloc(self, Py_ssize_t size)
    cpdef PinnedMemoryPointer malloc(self, Py_ssize_t size)
    cpdef PinnedMemoryPointer _malloc(self, Py_ssize_t size)
    cpdef free(self, size_t ptr, Py_ssize_t size)
    cpdef free_all_blocks(self)
    cpdef free_all_free(self)
    cpdef n_free_blocks(self)
    cpdef Py_ssize_t _round_size(self, Py_ssize_t size)
    cpdef int _bin_index_from_size(self, Py_ssize_t size)
    cpdef _append_to_free_list(self, Py_ssize_t size, chunk)
    cpdef bint _remove_from_free_list(self, Py_ssize_t size, chunk) except *
    cpdef tuple _split(self, PinnedChunk chunk, Py_ssize_t size)
    cpdef PinnedChunk _merge(self, PinnedChunk head, PinnedChunk remaining)