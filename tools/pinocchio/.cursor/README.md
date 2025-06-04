# Pinocchio - Choreo AI助手系统

Pinocchio是一个专门用于Choreo编程语言的AI辅助系统，它能够帮助开发者分析、生成、优化和调试Choreo代码。

## 功能特点

- **Choreo语法分析**：检测代码中的语法错误和潜在问题
- **代码生成**：根据需求生成符合Choreo语法的代码
- **性能优化**：为现有代码应用性能优化
- **调试辅助**：帮助定位和修复代码问题
- **TopSCC转Choreo**：辅助将TopSCC代码转换为Choreo代码

## 使用方法

当你需要Pinocchio系统的帮助时，只需在对话中提及"Pinocchio"，并描述你需要的任务，例如：

```
Pinocchio，请帮我检查以下Choreo代码中的语法错误：
[你的Choreo代码]
```

或者：

```
Pinocchio，请帮我将以下TopSCC代码转换为Choreo：
[你的TopSCC代码]
```

## 系统组件

Pinocchio系统由以下核心组件组成：

1. **协调器（Coordinator）**：管理其他组件的交互
2. **分析器（Analyzer）**：分析代码结构和识别问题
3. **生成器（Generator）**：生成符合规范的Choreo代码
4. **优化器（Optimizer）**：应用性能优化策略
5. **调试器（Debugger）**：帮助发现和修复问题

## Choreo语法规则

Pinocchio内置了丰富的Choreo语法规则知识，包括：

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

系统测试文件位于`tests/pinocchio/`目录下，包含：

- Choreo语法错误示例和正确示例
- 嵌入式缓存实现示例
- Choreo代码分析和翻译测试工具 