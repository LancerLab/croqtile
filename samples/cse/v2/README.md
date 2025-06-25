# Choreo CSE v2 Sample Notes

## Handling Device Buffers with Unknown Row Count

If `output_emb_vec` is a device buffer whose row count is completely unknown at compile time, you can refer to the implementation in `merge_emb_vec.co`:

- When constructing the `choreo::spanned_view` for the device buffer, simply specify a very large number (e.g., `UNKNOWN`) for the row dimension.
- In the kernel signature, use `?` to indicate the unknown dimension, e.g.:
  ```c++
  __co__ void merge_emb_vec_direct(..., global f32[?, EMB_VEC_SIZE] output_emb_vec)
  ```
- This allows the kernel to work with device buffers of unknown size, as long as the actual memory is large enough.

## Host Buffers with Unknown Row Count

For `fill_default_emb_vec.co` and `decompress_emb_vec.co`, the current implementation still passes host buffers (not device buffers) as input/output. If you have a host buffer with an unknown row count:

- You must allocate a fixed-size device buffer to handle the data.
- On the host side, write a loop to process the data in chunks, each time using `topsMemcpy` to transfer a fixed-size portion of the host buffer to the device buffer.
- The kernel can only process up to the fixed device buffer size per invocation.

### Why can't host buffers with unknown row size be fully handled as unknown dimension?

Even if the kernel signature uses `?` for the unknown dimension, the device buffer must be allocated with a concrete size before launching the kernel. The device memory allocator requires a specific number of rows to reserve the memory. If you do not specify a concrete size, the device may access out-of-bounds memory or trigger runtime errors. Therefore, for host buffers with unknown row size, you must always allocate a device buffer with a fixed (sufficiently large) size, and manage the data transfer and kernel invocation logic on the host side.

## Summary
- For device buffers: use a large shape in `spanned_view` and `?` in the kernel.
- For host buffers: allocate a fixed-size device buffer and manage chunking/copying on the host.

See `merge_emb_vec.co` for a working device buffer example. The other kernels will be updated to support device buffer input/output in the future.
