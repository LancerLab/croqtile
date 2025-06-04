#!/usr/bin/env python3
"""
Pinocchio - Choreo AI助手系统激活脚本
"""
import os
import json
import sys

def print_color(text, color="green"):
    """打印彩色文本"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['green'])}{text}{colors['reset']}")

def load_json_file(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def check_file_exists(file_path):
    """检查文件是否存在"""
    return os.path.isfile(file_path)

def check_directory_exists(dir_path):
    """检查目录是否存在"""
    return os.path.isdir(dir_path)

def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "pinocchio_config.json")
    
    print_color("\n🤖 Pinocchio - Choreo AI助手系统激活检查", "cyan")
    print_color("="*60, "cyan")
    
    # 检查配置文件
    if not check_file_exists(config_file):
        print_color("❌ 配置文件不存在:", "red")
        print_color(f"   {config_file}", "red")
        return
    
    # 加载配置文件
    config = load_json_file(config_file)
    if not config:
        print_color("❌ 配置文件加载失败", "red")
        return
    
    print_color(f"✅ 配置文件加载成功: {config['name']} v{config['version']}", "green")
    
    # 检查知识库目录
    knowledge_base_dir = os.path.join(script_dir, "knowledge")
    if not check_directory_exists(knowledge_base_dir):
        print_color(f"❌ 知识库目录不存在: {knowledge_base_dir}", "red")
        return
    
    # 检查规则目录
    rules_dir = os.path.join(script_dir, "rules")
    if not check_directory_exists(rules_dir):
        print_color(f"❌ 规则目录不存在: {rules_dir}", "red")
        return
    
    # 检查知识库文件
    knowledge_files = config.get("knowledge_files", {})
    print_color("\n📚 知识库文件检查:", "yellow")
    all_knowledge_files_exist = True
    
    for key, filename in knowledge_files.items():
        file_path = os.path.join(knowledge_base_dir, filename)
        if check_file_exists(file_path):
            print_color(f"  ✅ {key}: {filename}")
        else:
            print_color(f"  ❌ {key}: {filename} (不存在)", "red")
            all_knowledge_files_exist = False
    
    # 检查规则文件
    components = config.get("components", {})
    print_color("\n📜 规则文件检查:", "yellow")
    all_rule_files_exist = True
    
    for component, details in components.items():
        rule_file = details.get("rule_file")
        if rule_file:
            file_path = os.path.join(rules_dir, rule_file)
            if check_file_exists(file_path):
                print_color(f"  ✅ {component}: {rule_file}")
            else:
                print_color(f"  ❌ {component}: {rule_file} (不存在)", "red")
                all_rule_files_exist = False
    
    # 检查测试目录
    tests_dir = config.get("paths", {}).get("tests")
    if tests_dir and check_directory_exists(tests_dir):
        print_color(f"\n✅ 测试目录存在: {tests_dir}", "green")
    else:
        print_color(f"\n❌ 测试目录不存在: {tests_dir}", "red")
    
    # 总结
    print_color("\n📋 激活检查总结:", "cyan")
    if all_knowledge_files_exist and all_rule_files_exist:
        print_color("✅ Pinocchio系统已准备就绪！", "green")
        print_color("   可以通过在对话中提及\"Pinocchio\"来激活系统", "green")
        print_color("\n🔍 示例: \"Pinocchio，请帮我分析这段Choreo代码...\"", "blue")
    else:
        print_color("❌ Pinocchio系统未完全准备就绪，请检查上述错误", "red")

if __name__ == "__main__":
    main() 