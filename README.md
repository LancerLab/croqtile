# Pinocchio - Choreo AI助手系统

Pinocchio是一个专门为Choreo编程语言设计的AI辅助系统，它能够帮助开发者分析、生成、优化和调试Choreo代码。

## 功能特点

- **语法检查**: 自动检测Choreo代码中的语法错误和潜在问题
- **代码生成**: 根据需求生成符合Choreo语法的代码
- **代码转换**: 将TopSCC代码转换为Choreo代码
- **性能优化**: 为现有Choreo代码提供优化建议
- **调试辅助**: 帮助定位和修复Choreo代码中的问题

## 系统激活

Pinocchio系统默认已安装到`./.cursor/pinocchio/`目录下。您可以通过运行激活检查脚本来验证系统是否准备就绪：

```bash
./.cursor/pinocchio/activate_pinocchio.py
```

如果系统成功激活，您将看到成功消息。

## 使用方法

当您需要Pinocchio的帮助时，只需在对话中提及"Pinocchio"，并描述您的需求：

```
Pinocchio，请帮我分析这段Choreo代码...
```

或者：

```
Pinocchio，请将这段TopSCC代码转换为Choreo代码：
[您的TopSCC代码]
```

## Choreo语法规则

Pinocchio内置了完整的Choreo语法规则知识，包括：

- 数据类型必须使用小写名称（u32, s32, f32）
- 数组维度必须在单个方括号内用逗号分隔：`u32 [2, 4]`
- 数组元素访问必须使用`.at()`方法：`array.at(i, j)`
- 函数参数不能包含多维数组声明
- 变量必须先声明后赋值，不能在声明时赋值
- 逻辑表达式如`a && b`必须拆分为嵌套if语句
- 索引变量不支持算术运算
- 不支持十六进制常量
- parallel块和foreach循环必须包含实际操作

## 测试

如果您想测试Pinocchio系统的功能，可以使用`tests/pinocchio/`目录下的测试文件：

```bash
# 测试语法分析器
python tests/pinocchio/test_analyzer.py

# 测试代码转换器
python tests/pinocchio/test_translator.py
```

## 系统结构

- `./.cursor/pinocchio/`: 系统主目录
  - `knowledge/`: 知识库文件
  - `rules/`: 规则文件
  - `pinocchio_config.json`: 系统配置
  - `pinocchio_prompt.md`: 系统提示
  - `README.md`: 系统说明
  - `activate_pinocchio.py`: 激活检查脚本
- `./tests/pinocchio/`: 测试文件目录

## 如何贡献

如果您发现了新的Choreo语法规则或者有改进建议，请将它们添加到相应的知识库文件中：

- 语法规则: `./.cursor/pinocchio/knowledge/choreo_syntax_rules.json`
- 常见错误: `./.cursor/pinocchio/knowledge/choreo_common_errors.json`
- 转换规则: `./.cursor/pinocchio/knowledge/topscc_to_choreo_translation.json`

---

# Choreo - The C++ DSL for TileFlow Programming
Choreo is a low-level Embedded Domain Specific Language (**EDSL**) for C++ specifically engineered to program data movement entities like Direct-Memory-Accesses (DMA). 

Traditionally, programming DMA has focused on hardware configuration rather than the data itself. In modern heterogeneous hardware like GPUs, programmers often need to move smaller chunks of data to faster memory to enhance performance. This requirement can make programming more complex and sometimes results in hard-to-maintain code.

To address these challenges, Choreo is designed to simplify DMA programming by introducing a novel paradigm called **'TileFlow' programming**. It has already shown significant improvement on productivity, safety, and adaptibility over existing design, and is proven effective for building **high-performance machine learnig kernels** on heterogeneous hardware.

## Features and Design Targets
### Productivity
One of the standout features of Choreo by design is its ability of **mind-set saving** in **data tiling** tasks. This is achieved by introducing domain specific types, which simplify data **shape manipulation** to a level comparable to *Python*. For instance: 
```cpp
  f32 [8, 4, 12] shaped_data;
  new_shape : shaped_data.span { (0)/ 2, (1)/ 4, 1, (2)};
```
With this code, programmers can effortlessly create a shape with a tiling factor of {2, 4, 1} from data 'd' and even add an extra dimension to the 'new_shape', all in a single line. Compared with corresponding C++ code, which has to build array and apply trivial arithmetics, Choreo spares programmers from having to combine low-level abstractions. 

Furthermore, as Choreo simplify operations of data movement, it provides the high-level abstraction of tiled data movement:
```cpp
  dma.copy input.chunkat(tiling_factors) => shared;
```
This code moves a data chunk of 'input' with specified tiling factors to a storage location named 'shared'. The code is usually observed in programs with hardware DMA support. Choreo compiler hides the complexities of DMA configurations, index calculations, and storage management with easy-to-maintain semantics. Therefore, it allows programmers to concentrate on high-level strategies for building high-performance kernels, which are normally essential for building ML/HPC applications.

### Code Safety
Another primary design goal of Choreo is to **ensure code safety** by catching errors at compile-time or as early as possible at runtime. To achieve this, Choreo employs **compile-time checks** and instruments **runtime-check** based on the shapes and rules inferred from the *tileflow code*.

Bugs related to DMA are typically challenging to diagnose. However, with Choreo's safety checks, programmers can significantly reduce debugging efforts, thereby shortening the overall development cycle.

### Dynamic Shapes
Dynamic shape support is crucial for building many ML kernels. Choreo enhances the dynamic shape support via the **symbolic dimension** feature. Programmers can utilize the feature easily like the below code:

```
__co__ auto matmul(f32 [M, K] lhs, f32 [N, K] rhs) { ... }
```
'M', 'N' and 'K' are the symbolic shape dimensions. Programs program shaped inputs, such as tensors, in such a natural way. Such a design priors any existing systems (late 2024). Additionally, symbolic dimensions are also checked to ensure safety. As it is automatic, and systematic, it eliminates the need for non-systematic, explicitly programmed assertions by the users, thereby reducing boilerplate code.

### Visualization
**Analytic and visualization** is another compelling feature of Choreo, designed to help programmers understand tiling behaviors. For instance, consider the following data movement statement:

`f1 = dma.copy a.chunkat(p, x, y) => local;`

With Choreo's visualization capability, it renders figures like:

![visualizing the DMA statement](./images/simple_dma.png)

Programmers is easy to find the projection of the tiling and data movement behavior from this visualization. Such assistance can significantly reduce user erorrs when being properly used.

# Documentation for Reference
Consult the [Getting Started With Choreo](./Documents/Documentation/getting-started-with-choreo.md) to build and install Choreo.
Consult the [Choreo Tutorials](http://10.31.50.149:8000/) document for information on building Choreo and the detailed usage.

