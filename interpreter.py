# Triton-Puzzles-Lite interpreter patch, ported to triton >= 3.3 (tested on 3.7.1).
#
# Original file: https://github.com/SiriusNEO/Triton-Puzzles-Lite/blob/main/interpreter.py
# Reference:     https://github.com/Deep-Learning-Profiling-Tools/triton-viz
#
# Porting notes (3.1.0 -> 3.7.1):
#   * `RESERVED_KWS` was removed from triton.runtime.interpreter; modern GridExecutor
#     filters kwargs by function signature instead, so we drop the manual filtering.
#   * `_patch_lang` now returns a `_LangPatchScope` that must be `.restore()`d; the old
#     `importlib.reload(tl)` trick is gone.
#   * `GridExecutor.__call__` was rewritten upstream (pre_run_hooks, TensorWrapper,
#     TensorDescriptor, patch scope); our replacement mirrors the new body.
#   * `create_load`/`create_store` now delegate to the masked variants upstream, so
#     hooking `create_masked_load`/`create_masked_store` still covers every access.
#   * Only Grid / Load / Store records are consumed by collect_grid(), so the original
#     BinaryOp / MakeRange / ExpandDims / Dot / Reduce record types are dropped.
#   * Offset arithmetic now casts the uint64 pointer array to int64 explicitly, so
#     negative (underflowing) offsets are actually detected.

import inspect
import triton
import triton.language as tl
import numpy as np

from triton.runtime.interpreter import (
    GridExecutor,
    _implicit_cvt,
    interpreter_builder,
)
from triton.runtime.interpreter import _patch_lang as triton_patch_lang
from triton.runtime import JITFunction
from contextlib import contextmanager
from functools import wraps

## Records


class Tensor:
    __slots__ = ("ptr", "dtype", "stride", "shape", "element_size")

    def __init__(self, ptr, dtype, stride, shape, element_size):
        self.ptr = ptr
        self.dtype = dtype
        self.stride = stride
        self.shape = shape
        self.element_size = element_size


class Grid:
    __slots__ = ("idx",)

    def __init__(self, idx):
        self.idx = idx


class MemOp:
    __slots__ = ("ptr", "shape", "offsets", "access_masks", "invalid_access_masks",
                 "original_offsets", "original_masks")

    def __init__(self, ptr, shape, offsets, access_masks, invalid_access_masks,
                 original_offsets, original_masks):
        self.ptr = ptr
        self.shape = shape
        self.offsets = offsets
        self.access_masks = access_masks
        self.invalid_access_masks = invalid_access_masks
        self.original_offsets = original_offsets
        self.original_masks = original_masks


class Load(MemOp):
    pass


class Store(MemOp):
    pass


## Record builder

class RecordBuilder:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self._launches = []
        self._grid_idx = (0, 0, 0)
        self._grid_dim = (1, 1, 1)

    @property
    def launches(self):
        return self._launches

    def set_grid_dim(self, nx, ny, nz):
        self._grid_dim = (nx, ny, nz)
        self._launches.append({"grid": (nx, ny, nz), "tensors": [], "records": []})

    def set_grid_idx(self, x, y, z):
        self._grid_idx = (x, y, z)
        self._launches[-1]["records"].append(Grid((x, y, z)))

    def add_tensors(self, tensors):
        self._launches[-1]["tensors"].extend(tensors)

    def sort_tensor_handles(self):
        self._launches[-1]["tensors"].sort(key=lambda t: t.ptr)

    def get_tensor_ptr(self, ptr):
        tensors = self._launches[-1]["tensors"]
        ret = tensors[0]
        for t in tensors:
            if ptr < t.ptr:
                break
            ret = t
        return ret

    def add_record(self, record):
        self._launches[-1]["records"].append(record)


record_builder = RecordBuilder()


def _check_storage_contiguous(tensor):
    # Note: this is different from `tensor.is_contiguous()`.
    shape_prod = 1
    indices = sorted(range(len(tensor.stride())), key=tensor.stride().__getitem__)
    for i, index in enumerate(indices):
        stride = tensor.stride(index)
        shape = tensor.shape[index]
        if i == 0 and stride != 1:
            return False
        if i != 0 and stride != shape_prod:
            return False
        shape_prod *= shape
    return True


def _grid_executor_call(self, *args_dev, **kwargs):
    # Triton kernels cannot have keyword-only / varargs, so filtering by argspec.args
    # (as upstream does) is safe and replaces the old RESERVED_KWS handling.
    argspec = inspect.getfullargspec(self.fn)
    kwargs = {k: v for k, v in kwargs.items() if k in argspec.args}
    args_hst, kwargs_hst = self._init_args_hst(args_dev, kwargs)
    for hook in self.pre_run_hooks:
        hook(*args_hst, **kwargs_hst)
    patch_scope = triton_patch_lang(self.fn)
    try:
        args = inspect.getcallargs(self.fn, *args_hst, **kwargs_hst)
        # --- record the tensor table (needed to map raw pointers back to tensors) ---
        tensors = []
        for name, arg in args.items():
            if name in self.constexprs or not hasattr(arg, "data_ptr"):
                continue
            assert _check_storage_contiguous(
                arg
            ), "triton-viz only supports contiguously stored tensors for now"
            handle = _implicit_cvt(arg)
            tensors.append(
                Tensor(
                    handle.handle.data[0],
                    handle.dtype,
                    arg.stride(),
                    arg.shape,
                    arg.element_size(),
                )
            )
        call_args = {
            name: arg if name in self.constexprs else _implicit_cvt(arg)
            for name, arg in args.items()
        }
        call_args.pop("self", None)
        grid = self.grid(call_args) if callable(self.grid) else self.grid
        assert len(grid) <= 3, "grid must have at most 3 dimensions"
        grid = grid + (1,) * (3 - len(grid))
        interpreter_builder.set_grid_dim(*grid)
        record_builder.set_grid_dim(*grid)
        record_builder.add_tensors(tensors)
        record_builder.sort_tensor_handles()
        for x in range(grid[0]):
            for y in range(grid[1]):
                for z in range(grid[2]):
                    interpreter_builder.set_grid_idx(x, y, z)
                    record_builder.set_grid_idx(x, y, z)
                    self.fn(**call_args)
    finally:
        patch_scope.restore()
    self._restore_args_dev(args_dev, args_hst, kwargs, kwargs_hst)


def _jit_function_call(self, *args, **kwargs):
    triton_patch_lang(self.fn)
    return self.fn(*args, **kwargs)


## Hooks


def check_out_of_bounds_access(ptrs, masks):
    first_ptr = np.reshape(ptrs.data, (-1))[0]
    tensor_ptr = record_builder.get_tensor_ptr(first_ptr)
    offsets = ptrs.data.astype(np.int64) - int(tensor_ptr.ptr)
    max_valid_offset = int(np.prod(tensor_ptr.shape)) * tensor_ptr.element_size
    valid_access_masks = (offsets >= 0) & (offsets < max_valid_offset)
    invalid_access_masks = (~valid_access_masks) & masks.data.astype(bool)
    corrected_offsets = np.where(valid_access_masks, offsets, 0)
    return (
        tensor_ptr,
        valid_access_masks & masks.data.astype(bool),
        invalid_access_masks,
        corrected_offsets,
        offsets,
    )


def _create_masked_load(fn):
    @wraps(fn)
    def wrapper(ptrs, masks, other, cache_modifier, eviction_policy, is_volatile):
        (
            tensor_ptr,
            valid_access_masks,
            invalid_access_masks,
            corrected_offsets,
            original_offsets,
        ) = check_out_of_bounds_access(ptrs, masks)
        record_builder.add_record(
            Load(
                ptr=tensor_ptr.ptr,
                shape=ptrs.data.shape,
                offsets=corrected_offsets,
                access_masks=valid_access_masks,
                invalid_access_masks=invalid_access_masks,
                original_offsets=original_offsets,
                original_masks=masks.data,
            )
        )
        return fn(ptrs, masks, other, cache_modifier, eviction_policy, is_volatile)

    return wrapper


def _create_masked_store(fn):
    @wraps(fn)
    def wrapper(ptrs, value, masks, cache_modifier, eviction_policy):
        (
            tensor_ptr,
            valid_access_masks,
            invalid_access_masks,
            corrected_offsets,
            original_offsets,
        ) = check_out_of_bounds_access(ptrs, masks)
        record_builder.add_record(
            Store(
                ptr=tensor_ptr.ptr,
                shape=ptrs.data.shape,
                offsets=corrected_offsets,
                access_masks=valid_access_masks,
                invalid_access_masks=invalid_access_masks,
                original_offsets=original_offsets,
                original_masks=masks.data,
            )
        )
        # mask off invalid accesses so the interpreter does not write out of bounds
        return fn(ptrs, value, valid_access_masks, cache_modifier, eviction_policy)

    return wrapper


@contextmanager
def patch():
    old_grid_executor_call = GridExecutor.__call__
    old_jit_function_call = JITFunction.__call__
    old_create_masked_load = interpreter_builder.create_masked_load
    old_create_masked_store = interpreter_builder.create_masked_store
    GridExecutor.__call__ = _grid_executor_call
    JITFunction.__call__ = _jit_function_call
    interpreter_builder.create_masked_load = _create_masked_load(
        interpreter_builder.create_masked_load
    )
    interpreter_builder.create_masked_store = _create_masked_store(
        interpreter_builder.create_masked_store
    )
    try:
        yield
    finally:
        GridExecutor.__call__ = old_grid_executor_call
        JITFunction.__call__ = old_jit_function_call
        interpreter_builder.create_masked_load = old_create_masked_load
        interpreter_builder.create_masked_store = old_create_masked_store


## Collect


def collect_grid():
    for launch in record_builder.launches[-1:]:
        records, tensor_table, failures, access_offsets = collect_launch(launch)
    return records, tensor_table, failures, access_offsets


def collect_launch(launch):
    tensor_table = {t.ptr: (t, i) for i, t in enumerate(launch["tensors"])}
    failures = {}
    access_offsets = {}
    all_grids = {}
    last_grid = None
    program_records = []
    for r in launch["records"]:
        if isinstance(r, Grid):
            if last_grid is not None:
                all_grids[last_grid.idx] = program_records
                program_records = []
            last_grid = r
        program_records.append(r)
        if isinstance(r, (Store, Load)):
            access_offsets[last_grid.idx] = r.original_offsets
            if (r.invalid_access_masks & r.original_masks.astype(bool)).any():
                failures[last_grid.idx] = ~(
                    r.invalid_access_masks & r.original_masks.astype(bool)
                )
    all_grids[last_grid.idx] = program_records
    return all_grids, tensor_table, failures, access_offsets
