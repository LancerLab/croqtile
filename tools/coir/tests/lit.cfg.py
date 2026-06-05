import os
import lit.formats

config.name = 'CoIR'
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.dirname(__file__)

# The build directory is passed via --param or defaults to a common location.
build_dir = lit_config.params.get('build_dir',
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'build'))

coir_tools_dir = os.path.join(build_dir, 'tools', 'coir')

# LLVM tools (FileCheck, etc.)
llvm_tools_dir = lit_config.params.get('llvm_tools_dir',
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'extern',
                 'llvm-project', 'bin'))

config.substitutions.append(('%coir-opt',
    os.path.join(coir_tools_dir, 'coir-opt')))
config.substitutions.append(('%coir-gen',
    os.path.join(coir_tools_dir, 'coir-gen')))
config.substitutions.append(('%coir-codegen',
    os.path.join(coir_tools_dir, 'coir-codegen')))

# Add both coir tools and LLVM tools (FileCheck) to PATH
config.environment['PATH'] = os.pathsep.join([
    coir_tools_dir,
    llvm_tools_dir,
    config.environment.get('PATH', '')
])

config.excludes = ['Inputs', 'Output', 'CMakeLists.txt', 'lit.cfg.py']
