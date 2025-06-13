# Pinocchio 多Agent系统 for Choreo Copilot

- 目录结构：
  - knowledge/ 规则与知识库（如 choreo_syntax_rules.mdc）
  - .copilot/ agent配置、知识索引、README
- 支持 #pinocchio 指令一键唤起多agent协作，自动分析/调试/生成/优化choreo算子。
- agents.json 定义 choreo:analyzer、choreo:debugger、choreo:generator、choreo:optimizer、agent:coordinator 等角色。
- knowledge.json 统一管理知识库文件。
