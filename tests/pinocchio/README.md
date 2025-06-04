# Pinocchio测试套件

本目录包含Pinocchio（Choreo AI助手系统）的测试用例和测试工具。

## 测试文件说明

- `test_choreo_errors.choreo`：包含常见Choreo语法错误的测试用例
- `test_choreo_correct.choreo`：遵循Choreo语法规则的正确代码示例
- `embedding_cache.choreo`：嵌入式缓存的Choreo实现示例
- `test_analyzer.py`：Choreo代码分析工具，用于测试语法规则检查功能
- `test_translator.py`：TopSCC到Choreo的转换工具，用于测试代码转换功能

## 运行测试

### 语法检查测试
```bash
python test_analyzer.py
```

### 代码转换测试
```bash
python test_translator.py
```

## 测试模式

当您想测试Pinocchio系统时，可以使用以下格式的提示：

```
Pinocchio，请分析tests/pinocchio/test_choreo_errors.choreo文件中的语法错误
```

或者：

```
Pinocchio，请将以下TopSCC代码转换为Choreo：
[TopSCC代码示例]
```

## 预期结果

- **分析测试**：Pinocchio应该能够识别出`test_choreo_errors.choreo`中的所有语法错误，并提供修复建议
- **转换测试**：Pinocchio应该能够正确将TopSCC代码转换为符合Choreo语法规则的代码
- **正确性验证**：Pinocchio应该能够验证`test_choreo_correct.choreo`文件不包含任何语法错误 