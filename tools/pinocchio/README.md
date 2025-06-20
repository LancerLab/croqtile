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

