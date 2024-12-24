## Issue Title: [Short description of the issue]

### Description
[Briefly describe the issue you're experiencing, and the context around it.]
For example: The issue seems to be related to the `dma.copy` in the `global` scope, causing an assertion failure during execution.

### Code to Reproduce
[Steps to reproduce the bug]
For example: Create a file (e.g., `example.co`) with the following code:
    ```co
    __co__ auto [function_name]([parameter_type] [array_dimension] [variable_name1], [parameter_type] [array_dimension] [variable_name2], [parameter_type] [variable_name3]) {
      f = dma.any;
      f = dma.copy [variable_name1] => global;  // This line might cause the issue
      parallel p by 1 {
        d = dma.any;
        with tiling in [8, 2] {
          d = dma.copy [variable_name1].chunkat(tiling) => local;
        }
      }
    }
    ```

### Commands to Reproduce
Run the following command to reproduce the issue:
    ```bash
    choreo [file_name]
    ```

### Unexpected Output
[Describe what you expected the program to do.]
For example: The program should run successfully and generate the correct DMA buffer for the `global` scope without errors.

### Possible Cause
[If you have an idea about what might be causing the issue, describe it here.]
For example: The issue may be related to how DMA buffers are being handled in the `global` scope, and could be triggered during the late normalization phase.
