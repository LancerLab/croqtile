#!/usr/bin/env python3
import json
import os
import sys

def load_json_file(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def read_choreo_file(file_path):
    """读取Choreo文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def analyze_choreo_code(code, rules):
    """分析Choreo代码，找出常见错误"""
    issues = []
    
    # 检查数据类型错误
    for line in code.splitlines():
        line = line.strip()
        
        # 检查大写类型名
        if any(type_name in line for type_name in ["U32", "S32", "F32", "U64", "S64", "F64"]):
            issues.append({
                "type": "数据类型错误",
                "description": "使用了大写类型名",
                "line": line,
                "solution": "使用小写类型名: u32, s32, f32"
            })
        
        # 检查多维数组声明错误
        if "] [" in line and "u32" in line.lower():
            issues.append({
                "type": "数组声明错误",
                "description": "使用了多个方括号声明多维数组",
                "line": line,
                "solution": "使用单个方括号和逗号分隔维度: u32 [2, 4] array;"
            })
        
        # 检查数组访问错误
        if "[" in line and "]" in line and "=" in line and ".at(" not in line:
            if "array" in line or "cache" in line:
                issues.append({
                    "type": "数组访问错误",
                    "description": "使用方括号访问数组元素",
                    "line": line,
                    "solution": "使用.at()方法: array.at(i, j)"
                })
        
        # 检查变量声明时赋值
        if "=" in line and any(type_name in line for type_name in ["u32", "s32", "f32", "U32", "S32", "F32"]):
            if ";" in line and ".at(" not in line:
                issues.append({
                    "type": "变量初始化错误",
                    "description": "变量声明时赋值",
                    "line": line,
                    "solution": "分开声明和赋值: u32 var; var = 123;"
                })
        
        # 检查复合逻辑表达式
        if "&&" in line or "||" in line:
            issues.append({
                "type": "逻辑表达式错误",
                "description": "使用了复合逻辑表达式",
                "line": line,
                "solution": "使用嵌套if语句: if (a) { if (b) { ... } }"
            })
        
        # 检查浮点数后缀
        if "0." in line and "f" not in line and "f32" in line.lower():
            issues.append({
                "type": "浮点字面量错误",
                "description": "浮点数缺少f后缀",
                "line": line,
                "solution": "添加f后缀: 0.5f"
            })
        
        # 检查索引变量算术运算
        if "+" in line and "[" in line and "]" in line:
            issues.append({
                "type": "索引算术错误",
                "description": "索引变量使用算术运算",
                "line": line,
                "solution": "使用中间变量: idx = i; idx = idx + 1;"
            })
        
        # 检查十六进制常量
        if "0x" in line:
            issues.append({
                "type": "常量表示错误",
                "description": "使用十六进制常量",
                "line": line,
                "solution": "使用十进制表示: 255 // 代替0xFF"
            })
    
    # 检查parallel和foreach结构
    if "parallel" in code:
        if "parallel p by" in code and "with" not in code:
            issues.append({
                "type": "parallel结构错误",
                "description": "parallel块没有with子句",
                "line": "parallel p by ...",
                "solution": "添加with子句: parallel p by 1 { with i in [10] { ... } }"
            })
    
    if "foreach" in code and "with" not in code:
        issues.append({
            "type": "foreach结构错误",
            "description": "foreach没有前置with子句",
            "line": "foreach i { ... }",
            "solution": "添加with子句: with i in [10] { foreach i { ... } }"
        })
    
    return issues

def print_analysis_report(file_path, issues):
    """打印分析报告"""
    print(f"\n{'='*80}")
    print(f"Choreo代码分析报告: {file_path}")
    print(f"{'='*80}")
    
    if not issues:
        print("✅ 没有发现问题，代码遵循Choreo语法规则！")
        return
    
    print(f"发现 {len(issues)} 个问题:\n")
    
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue['type']}")
        print(f"   问题: {issue['description']}")
        print(f"   代码: {issue['line']}")
        print(f"   解决方案: {issue['solution']}")
        print()

def main():
    """主函数"""
    # 更新知识库路径，指向根目录的Pinocchio系统
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    knowledge_dir = os.path.join(project_root, ".cursor/pinocchio/knowledge")
    rules_file = os.path.join(knowledge_dir, "choreo_syntax_rules.json")
    errors_file = os.path.join(knowledge_dir, "choreo_common_errors.json")
    
    # 加载规则文件
    rules = load_json_file(rules_file)
    errors = load_json_file(errors_file)
    
    if not rules or not errors:
        print("Failed to load rules or errors knowledge base")
        return
    
    # 测试文件目录
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 获取要分析的文件
    if len(sys.argv) > 1:
        files_to_analyze = sys.argv[1:]
    else:
        files_to_analyze = [
            os.path.join(test_dir, "test_choreo_errors.choreo"),
            os.path.join(test_dir, "test_choreo_correct.choreo"),
            os.path.join(test_dir, "embedding_cache.choreo")
        ]
    
    # 分析每个文件
    for file_path in files_to_analyze:
        code = read_choreo_file(file_path)
        if code:
            issues = analyze_choreo_code(code, rules)
            print_analysis_report(file_path, issues)

if __name__ == "__main__":
    main() 