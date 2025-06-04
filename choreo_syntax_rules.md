# Choreo语法规则知识库

## 基本数据类型和索引
1. Choreo中索引使用`int`类型，而非`u32`或其他无符号类型
2. 数据类型包括`f32`(浮点数)、`u32`(无符号整数)等，但在索引计算中应使用`int`

## 并行结构
1. `parallel p by N`：定义并行执行块，其中`N`是并行度，`p`是并行ID
2. 可以访问`p.id`获取当前并行实例的ID（从0开始）
3. 并行块可以嵌套，例如`parallel p by 2, q by 8`

## 循环和遍历
1. `with`语句用于定义有界变量，例如`with i in [N]`
2. `foreach`语句用于遍历有界变量，例如`foreach i in [N]`
3. **正确用法**：先使用`with`定义变量范围，然后在循环体内使用该变量
   ```choreo
   with i in [N] {
     // 使用i进行操作
   }
   ```
4. **错误用法**：不应直接在`foreach`中使用`inthreads`
   ```choreo
   // 错误：不要这样使用
   foreach i in [N + inthreads(M)] { ... }
   ```

## 条件执行 - inthreads
1. `inthreads(条件表达式)`：用于条件执行，类似于if语句
2. 条件表达式必须是严格发散的（与并行ID相关）
3. `inthreads`块结束后会自动同步所有线程
4. `inthreads.async`用于异步条件执行，不会自动同步

## 正确的inthreads用法
1. 在并行块内使用：
   ```choreo
   parallel p by N {
     inthreads(p < M) {
       // 只有ID小于M的线程执行
     }
   }
   ```
2. 用于边界检查：
   ```choreo
   int idx = base + p.id;
   inthreads(idx < max_bound) {
     // 安全访问，避免越界
   }
   ```
3. 与with语句配合使用（正确方式）：
   ```choreo
   with i in [N] {
     int idx = i * stride + p.id;
     inthreads(idx < bound) {
       // 安全操作
     }
   }
   ```
4. inthreads与foreach连用（单行语法）：
   ```choreo
   // 条件遍历语法
   inthreads(p == 0) foreach q { ... }  // 只有p.id为0的线程执行foreach循环
   inthreads(p < N % #p) foreach x, y { ... }  // 处理余数情况
   ```

## 内存访问
1. 使用`.at()`进行数组访问，例如`array.at(i, j)`
2. 使用`.span`获取数组维度
3. 使用`.chunkat()`进行数据块访问

## 事件和同步
1. `shared event e`：定义共享事件
2. `trigger e`：触发事件
3. `wait e`：等待事件
4. `sync.shared`：同步共享内存

## 硬件资源限制

### GCU硬件资源（图形计算单元）
1. **GCU 3代**：
   - Cluster数量最大为2
   - 每个Cluster内SIP数量最大为12
   - 每个Cluster的Thread数量通常为8
   - 设置`parallel by`参数时应考虑实际硬件资源限制

### GPU硬件资源
1. **通用GPU**：
   - SM (Streaming Multiprocessor)数量根据型号不同而变化
   - 每个SM包含多个CUDA核心
   - 线程组织为warp（通常32线程）
   - 共享内存大小有限，需要合理管理 