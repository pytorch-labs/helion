# pyre-ignore-all-errors[2] # ignore Missing parameter annotation
from __future__ import annotations

import triton
import triton.language as tl

__all__ = ["_triton_send_signal", "_triton_wait_multiple_signal", "_triton_wait_signal"]


@triton.jit
def _triton_send_signal(
    addr,  # can be a scalar or a vector of pointers.
    update: tl.constexpr,
    sem: tl.constexpr,
    scope: tl.constexpr,
    op: tl.constexpr,
    skip_sync: tl.constexpr,
) -> None:
    """
    Send a signal to a global memory barrier.

    This function implements a spin-wait loop that continuously checks a memory location
    until it reaches the expected value, providing synchronization across GPU threads.

    Args:
        addr: Memory address of the barrier to wait on (Must be a scalar)
        expect: Expected value to wait for
        update: Update
    """
    if not skip_sync:
        tl.inline_asm_elementwise(
            "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
        )

    tl.static_assert(
        sem == "release" or sem == "relaxed",
        "Invalid memory semantic. options: 'release', 'relaxed'. ",
    )
    tl.static_assert(
        scope == "gpu" or scope == "sys", "Invalid scope. options: 'gpu','sys'. "
    )

    if op == "atomic_xchg":
        tl.atomic_xchg(addr, update, sem=sem, scope=scope)
    elif op == "atomic_add":
        tl.atomic_add(addr, update, sem=sem, scope=scope)
    else:
        raise NotImplementedError(
            f"Unsupported op '{op}' for send signal on gmem barrier. "
        )


@triton.jit
def _triton_wait_signal(
    addr,
    expect: tl.constexpr,  # wait until lock is set to expect
    update: tl.constexpr,  # update the lock once it is aquired.
    sem: tl.constexpr,
    scope: tl.constexpr,
    op: tl.constexpr,
    skip_sync: tl.constexpr,
) -> None:
    """
    Wait for a global memory barrier to reach the expected state.

    This function implements a spin-wait loop that continuously checks a memory location
    until it reaches the expected value, providing synchronization across GPU threads.

    Args:
        addr: Memory address of the barrier to wait on (Must be a scalar)
        expect: Expected value to wait for
        update: Update the barrier with once acquired
        sem: Memory semantics for the atomic operation. Options: "acquire", "relaxed".
        scope: Scope of the atomic operation. Options: "gpu", "sys"
        op: Atomic operation type: "ld", "atomic_cas"
    """
    tl.static_assert(
        addr.type.is_ptr(),
        "Barrier address must be a scalar. Do you want to use '_triton_wait_multiple_signal'? ",
    )

    tl.static_assert(
        sem == "acquire" or sem == "relaxed",
        "Invalid memory semantic. options: 'acquire', 'relaxed'. ",
    )
    tl.static_assert(
        scope == "gpu" or scope == "sys", "Invalid scope. options: 'gpu', 'sys'. "
    )
    tl.static_assert(
        op == "ld" or op == "atomic_cas",
        "Invalid op. options: 'ld', 'atomic_cas'. ",
    )

    # Spin-wait loop:
    #   Uses atomic_add with update=0 for ld.global.{sem}.{scope}
    #   Triton generates smem broadcasting of tl.atomic_add return value in ptx,
    #   but it is optimized away by ptxas in SASS, hence no performance overhead.
    if op == "ld":
        tl.static_assert(
            update == 0, "ld wait on gmem_barriers cannot update the lock. "
        )
        while tl.atomic_add(addr, 0, sem=sem, scope=scope) != expect:
            pass
    elif op == "atomic_cas":
        while tl.atomic_cas(addr, expect, update, sem=sem, scope=scope) != expect:
            pass
    else:
        raise NotImplementedError(
            f"Unsupported op '{op}' for wait signal on gmem barrier. "
        )

    if not skip_sync:
        tl.inline_asm_elementwise(
            "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
        )
    # tl.debug_barrier() cause significant performance loss. (Perhaps breaks triton prefetching?)


@triton.jit
def _triton_wait_multiple_signal(
    addr,
    expect: tl.constexpr,  # wait until lock is set to expect
    update: tl.constexpr,  # update the lock once it is aquired.
    sem: tl.constexpr,
    scope: tl.constexpr,
    op: tl.constexpr,
    skip_sync: tl.constexpr,
) -> None:
    raise NotImplementedError("Waiting on multiple barriers is not implemented yet. ")
    # TODO(joydddd): waiting on multiple barriers at the same time whereeach thread waits on a different barrier
