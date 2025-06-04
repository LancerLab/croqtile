#!/usr/bin/env python3
import json
import os
import sys
import re

def load_json_file(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def read_file(file_path):
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def translate_topscc_to_choreo(topscc_code, translation_rules):
    """将TopSCC代码转换为Choreo代码"""
    choreo_code = topscc_code
    
    # 应用数据类型转换
    for type_rule in translation_rules["data_types"]["basic_types"]:
        choreo_code = re.sub(r'\b' + type_rule["topscc"] + r'\b', type_rule["choreo"], choreo_code)
    
    # 应用浮点字面量转换
    for literal_rule in translation_rules["data_types"]["literals"]:
        if "0x" not in literal_rule["topscc"]:  # 排除十六进制，它需要特殊处理
            choreo_code = choreo_code.replace(literal_rule["topscc"], literal_rule["choreo"])
    
    # 应用十六进制转换（简单实现，实际需要更复杂的处理）
    hex_pattern = r'0x([0-9A-Fa-f]+)'
    for match in re.finditer(hex_pattern, choreo_code):
        hex_value = match.group(0)
        dec_value = str(int(hex_value, 16))
        choreo_code = choreo_code.replace(hex_value, dec_value)
    
    # 简单处理数组声明转换（这是一个简化版，实际需要更复杂的处理）
    # 将 Type array[x][y]; 转换为 type [x, y] array;
    array_pattern = r'(\w+)\s+(\w+)\[(\d+)\]\[(\d+)\];'
    for match in re.finditer(array_pattern, choreo_code):
        type_name, var_name, dim1, dim2 = match.groups()
        original = match.group(0)
        # 转换为Choreo格式
        if type_name.isupper():
            type_name = type_name.lower()
        replacement = f"{type_name} [{dim1}, {dim2}] {var_name};"
        choreo_code = choreo_code.replace(original, replacement)
    
    # 转换数组访问
    # 将 array[i][j] 转换为 array.at(i, j)
    access_pattern = r'(\w+)\[(\w+)\]\[(\w+)\]'
    for match in re.finditer(access_pattern, choreo_code):
        array_name, idx1, idx2 = match.groups()
        original = match.group(0)
        replacement = f"{array_name}.at({idx1}, {idx2})"
        choreo_code = choreo_code.replace(original, replacement)
    
    # 单维数组访问
    access_pattern_1d = r'(\w+)\[(\w+)\]'
    for match in re.finditer(access_pattern_1d, choreo_code):
        array_name, idx = match.groups()
        original = match.group(0)
        # 排除已经转换的多维数组访问
        if ".at(" not in original:
            replacement = f"{array_name}.at({idx})"
            choreo_code = choreo_code.replace(original, replacement)
    
    # 变量声明和赋值分离
    # 将 Type var = value; 转换为 Type var; var = value;
    var_init_pattern = r'(\w+)\s+(\w+)\s*=\s*([^;]+);'
    for match in re.finditer(var_init_pattern, choreo_code):
        type_name, var_name, value = match.groups()
        original = match.group(0)
        if type_name.lower() in ["u32", "s32", "f32", "u8", "s8", "u16", "s16"]:
            replacement = f"{type_name} {var_name};\n{var_name} = {value};"
            choreo_code = choreo_code.replace(original, replacement)
    
    # 转换逻辑表达式
    # 注意：这是一个极度简化的版本，实际需要更复杂的解析
    logic_pattern = r'if\s*\(([^&|]+)&&([^)]+)\)'
    for match in re.finditer(logic_pattern, choreo_code):
        cond1, cond2 = match.groups()
        original = match.group(0)
        replacement = f"if ({cond1.strip()}) {{\n  if ({cond2.strip()})"
        choreo_code = choreo_code.replace(original, replacement)
    
    # 简单替换 kernel 函数定义
    choreo_code = re.sub(r'__global__\s+void', '__co__ void', choreo_code)
    
    return choreo_code

def create_topscc_sample():
    """创建一个TopSCC示例代码"""
    return """// TopSCC嵌入式缓存示例
#define CACHE_SIZE 1024
#define SET_ASSOCIATIVITY 8

// 缓存结构
U32 cache_keys[SET_ASSOCIATIVITY][CACHE_SIZE];
U32 cache_values[SET_ASSOCIATIVITY][CACHE_SIZE];
U32 cache_valid[SET_ASSOCIATIVITY][CACHE_SIZE];

// 初始化缓存
void init_cache() {
    for (U32 i = 0; i < SET_ASSOCIATIVITY; i++) {
        for (U32 j = 0; j < CACHE_SIZE; j++) {
            cache_keys[i][j] = 0;
            cache_values[i][j] = 0;
            cache_valid[i][j] = 0;
        }
    }
}

// 查找缓存
U32 lookup_cache(U32 key) {
    U32 set_idx = key % CACHE_SIZE;
    U32 found = 0;
    U32 result = 0;
    
    for (U32 i = 0; i < SET_ASSOCIATIVITY; i++) {
        if (cache_valid[i][set_idx] != 0 && cache_keys[i][set_idx] == key) {
            found = 1;
            result = cache_values[i][set_idx];
        }
    }
    
    return result;
}

// 更新缓存
void update_cache(U32 key, U32 value) {
    U32 set_idx = key % CACHE_SIZE;
    U32 replaced = 0;
    
    // 先查找键是否已存在
    for (U32 i = 0; i < SET_ASSOCIATIVITY; i++) {
        if (cache_keys[i][set_idx] == key) {
            cache_values[i][set_idx] = value;
            cache_valid[i][set_idx] = 1;
            replaced = 1;
        }
    }
    
    // 查找空槽位
    if (replaced == 0) {
        for (U32 i = 0; i < SET_ASSOCIATIVITY; i++) {
            if (cache_valid[i][set_idx] == 0 && replaced == 0) {
                cache_keys[i][set_idx] = key;
                cache_values[i][set_idx] = value;
                cache_valid[i][set_idx] = 1;
                replaced = 1;
            }
        }
    }
    
    // 如果没有空槽位，替换第一个
    if (replaced == 0) {
        cache_keys[0][set_idx] = key;
        cache_values[0][set_idx] = value;
        cache_valid[0][set_idx] = 1;
    }
}

// CUDA kernel示例
__global__ void process_embeddings(U32* keys, U32* values, U32 count) {
    U32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        U32 key = keys[idx];
        U32 value = lookup_cache(key);
        if (value == 0) {
            value = 0xFF; // 使用默认值
            update_cache(key, value);
        }
        values[idx] = value;
    }
}"""

def main():
    """主函数"""
    # 更新知识库路径，指向根目录的Pinocchio系统
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    knowledge_dir = os.path.join(project_root, ".cursor/pinocchio/knowledge")
    translation_file = os.path.join(knowledge_dir, "topscc_to_choreo_translation.json")
    
    # 加载转换规则
    translation_rules = load_json_file(translation_file)
    
    if not translation_rules:
        print("Failed to load translation rules")
        return
    
    # 创建TopSCC示例代码
    topscc_code = create_topscc_sample()
    
    # 转换为Choreo代码
    choreo_code = translate_topscc_to_choreo(topscc_code, translation_rules)
    
    # 打印结果
    print("\n" + "="*80)
    print("TopSCC 原始代码:")
    print("="*80)
    print(topscc_code)
    
    print("\n" + "="*80)
    print("转换后的 Choreo 代码:")
    print("="*80)
    print(choreo_code)

if __name__ == "__main__":
    main() 