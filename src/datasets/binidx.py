from lib2to3.pgen2 import token
import os
import torch
import numpy as np
import shutil
import struct
from functools import lru_cache
from itertools import accumulate


def print_rank_0(*message):
    pass


def _warmup_mmap_file(path):
    pass


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: float,
    7: np.double,
    8: np.uint16,
}

NEW_DTYPES = {
    0: np.uint8,
    1: np.uint16,
    2: np.float32,
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"


class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, "wb")

                    self._file.write(cls._HDR_MAGIC)

                    self._file.write(struct.pack("<Q", 1))

                    self._file.write(struct.pack("<B", code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes, doc_idx):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack("<Q", len(sizes)))

                    self._file.write(struct.pack("<Q", len(doc_idx)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order="C"))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, "rb") as stream:
                magic_test = stream.read(len(self._HDR_MAGIC))
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )

                version_bytes = stream.read(8)
                self._version = struct.unpack("<Q", version_bytes)[0]

                if self._version == 1:

                    (dtype_code,) = struct.unpack("<B", stream.read(1))
                    self._dtype = dtypes[dtype_code]
                    self._token_unit_len = 1
                elif self._version == 2:

                    (token_unit_type_code,) = struct.unpack("<B", stream.read(1))

                    self._token_unit_len = struct.unpack("<I", stream.read(4))[0]
                    if self._token_unit_len == 0:
                        raise ValueError("token_unit_len cannot be zero in version 2 format.")

                    print_rank_0(
                        f"[MMapIndexedDataset.Index V2] token_unit_type_code from file: {token_unit_type_code}")
                    print_rank_0(f"[MMapIndexedDataset.Index V2] token_unit_len from file: {self._token_unit_len}")

                    if token_unit_type_code in NEW_DTYPES:
                        self._dtype = NEW_DTYPES[token_unit_type_code]
                    elif token_unit_type_code in dtypes:
                        print_rank_0(
                            f"Warning: token_unit_type_code {token_unit_type_code} not in NEW_DTYPES, falling back to legacy dtypes.")
                        self._dtype = dtypes[token_unit_type_code]
                    else:
                        raise ValueError(
                            f"Unknown token_unit_type_code {token_unit_type_code} for version 2 index. "
                            "Please update NEW_DTYPES in binidx.py."
                        )
                else:
                    raise ValueError(f"Unknown index version {self._version}")

                print_rank_0(
                    f"[MMapIndexedDataset.Index] Version: {self._version}, Resolved Dtype: {self._dtype}, Token Unit Len: {self._token_unit_len if hasattr(self, '_token_unit_len') else 'N/A (V1)'}")

                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                self._doc_count = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                print_rank_0("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            print_rank_0("    reading sizes...")

            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
            )

            pointers_offset = offset + self._sizes.nbytes
            if self._version == 1:
                pointers_count = self._len
            elif self._version == 2:
                pointers_count = self._doc_count
            else:
                raise ValueError(f"Unsupported version {self._version} for pointer count.")

            print_rank_0("    reading pointers...")
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=pointers_count,
                offset=pointers_offset,
            )

            doc_idx_offset = pointers_offset + self._pointers.nbytes
            print_rank_0("    reading document index...")

            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=doc_idx_offset,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(
            data_file_path(self._path), mode="r", order="C"
        )
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, logical_size = self._index[idx]
            if self._index._version == 1:
                elements_to_read = logical_size
            elif self._index._version == 2:
                elements_to_read = logical_size * self._index._token_unit_len
            else:
                raise ValueError(f"Unknown index version {self._index._version}")

            if elements_to_read == 0:
                if self._index._token_unit_len == 1:
                    return np.empty((0,), dtype=self._index.dtype)
                else:
                    return np.empty((0, self._index._token_unit_len), dtype=self._index.dtype)

            np_array_flat = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=elements_to_read, offset=ptr
            )

            if np_array_flat.size > 0:
                if np_array_flat.size % self._index._token_unit_len != 0:
                    raise ValueError(
                        f"Data size {np_array_flat.size} is not compatible with token_unit_len {self._index._token_unit_len} for reshaping."
                    )

                if self._index._token_unit_len == 1:
                    return np_array_flat
                else:
                    return np_array_flat.reshape(logical_size, self._index._token_unit_len)
            else:
                if self._index._token_unit_len == 1:
                    return np.empty((0,), dtype=self._index.dtype)
                else:
                    return np.empty((0, self._index._token_unit_len), dtype=self._index.dtype)

        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError(
                    "Slices into indexed_dataset must be contiguous")

            if start >= stop:
                return []

            ptr = self._index._pointers[start]

            logical_sizes_in_slice = self._index._sizes[idx]

            if self._index._version == 1:
                total_elements_to_read = sum(logical_sizes_in_slice)
                split_offsets_flat = list(accumulate(logical_sizes_in_slice))
            elif self._index._version == 2:
                total_elements_to_read = sum(s * self._index._token_unit_len for s in logical_sizes_in_slice)
                split_offsets_flat = list(accumulate(s * self._index._token_unit_len for s in logical_sizes_in_slice))
            else:
                raise ValueError(f"Unknown index version {self._index._version}")

            if total_elements_to_read == 0:
                empty_shape = (0,) if self._index._token_unit_len == 1 else (0, self._index._token_unit_len)
                return [
                    np.empty(empty_shape, dtype=self._index.dtype)
                    for _ in logical_sizes_in_slice
                ]

            np_array_bulk_flat = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=total_elements_to_read, offset=ptr
            )

            if not split_offsets_flat or not split_offsets_flat[:-1]:
                if total_elements_to_read > 0:
                    sents_flat = [np_array_bulk_flat]
                else:
                    sents_flat = []
            else:
                sents_flat = np.split(np_array_bulk_flat, split_offsets_flat[:-1])

            reshaped_sents = []
            for i, item_flat_array in enumerate(sents_flat):
                current_item_logical_size = logical_sizes_in_slice[i]
                if item_flat_array.size > 0:
                    if item_flat_array.size % self._index._token_unit_len != 0:
                        raise ValueError(
                            f"Item data size {item_flat_array.size} in slice is not compatible with token_unit_len {self._index._token_unit_len} for reshaping."
                        )

                    if self._index._token_unit_len == 1:
                        reshaped_sents.append(item_flat_array)
                    else:
                        reshaped_sents.append(
                            item_flat_array.reshape(current_item_logical_size, self._index._token_unit_len)
                        )
                elif current_item_logical_size == 0:
                    empty_shape = (0,) if self._index._token_unit_len == 1 else (0, self._index._token_unit_len)
                    reshaped_sents.append(np.empty(empty_shape, dtype=self._index.dtype))
                else:
                    empty_shape = (0,) if self._index._token_unit_len == 1 else (0, self._index._token_unit_len)
                    reshaped_sents.append(np.empty(empty_shape, dtype=self._index.dtype))

            return reshaped_sents
        return None

    def get(self, idx, offset=0, length=0):
        """Retrieves an item or a portion of an item from the dataset, with 'free addressing'.

        The \`offset\` and \`length\` parameters are in units of **logical tokens**.
        \'idx\` is used to get the starting byte pointer of the item/stream.
        The size of the item specified by \`idx\` in the .idx file does NOT cap the read length.
        Reading beyond the end of the mmaped .bin file will result in an error from np.frombuffer.

        The returned array will have shape \`[num_logical_tokens_read, token_unit_len]\`.
        """
        item_start_byte_ptr, _ = self._index[idx]

        token_unit_len = self._index._token_unit_len
        dtype_byte_size = self._index._dtype_size

        final_offset_logical_tokens = max(0, offset if offset is not None else 0)
        num_logical_tokens_to_read = max(0, length if length is not None else 0)

        start_read_byte_offset = item_start_byte_ptr + (final_offset_logical_tokens * token_unit_len * dtype_byte_size)

        num_base_elements_to_read = num_logical_tokens_to_read * token_unit_len

        np_array_flat = np.frombuffer(
            self._bin_buffer,
            dtype=self._index.dtype,
            count=num_base_elements_to_read,
            offset=start_read_byte_offset
        )

        return np_array_flat.reshape(num_logical_tokens_to_read, token_unit_len)

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(
            data_file_path(path)
        )
