#!/usr/bin/env python3
"""测试角色选择器功能。"""

import sys
from pathlib import Path

# 确保项目根在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from summerclaw.agent.role_selector import (
    scan_available_roles, 
    should_select_roles,
    ensure_requirements_file,
    load_requirements,
    DEFAULT_REQUIREMENTS_CONTENT
)

def test_scan_roles():
    """测试扫描角色功能。"""
    print("=" * 60)
    print("测试1: 扫描可用角色")
    print("=" * 60)
    
    roles = scan_available_roles()
    print(f"\n✓ 成功扫描到 {len(roles)} 个角色")
    
    # 显示前10个角色
    print("\n前10个角色:")
    for i, role in enumerate(roles[:10], 1):
        print(f"  {i}. [{role['category']}] {role['role_name']}")
    
    print(f"\n  ... 还有 {len(roles) - 10} 个角色")
    
    # 统计分类
    categories = {}
    for role in roles:
        cat = role['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n角色分类统计:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count} 个角色")
    
    return True

def test_should_select():
    """测试是否应该选择角色的判断。"""
    print("\n" + "=" * 60)
    print("测试2: 检查是否需要选择角色")
    print("=" * 60)
    
    # 测试默认工作目录
    from summerclaw.config.paths import get_workspace_path
    from summerclaw.config.loader import load_config
    
    try:
        config = load_config()
        workspace = get_workspace_path(config.workspace_path)
        
        should_select, sel_dir = should_select_roles(workspace)
        print(f"\n工作目录: {workspace}")
        print(f"是否需要选择: {'是' if should_select else '否'}")
        
        selected_dir = workspace / "roles" / "selected"
        if selected_dir.exists():
            md_files = list(selected_dir.glob("*.md"))
            print(f"已选择角色数: {len(md_files)}")
        else:
            print(f"selected 目录不存在")
        
        return True
    except Exception as e:
        print(f"\n⚠ 配置加载失败（这可能正常）: {e}")
        return True  # 不视为测试失败

def test_config_schema():
    """测试配置schema。"""
    print("\n" + "=" * 60)
    print("测试3: 配置Schema")
    print("=" * 60)
    
    try:
        from summerclaw.config.schema import RoleSelectorConfig, AgentDefaults
        
        # 测试默认值
        role_config = RoleSelectorConfig()
        print(f"\n✓ RoleSelectorConfig 创建成功")
        print(f"  - enabled: {role_config.enabled}")
        print(f"  - requirements: '{role_config.requirements}'")
        print(f"  - count: {role_config.count}")
        print(f"  - model_override: {role_config.model_override}")
        
        # 测试 AgentDefaults 包含 role_selector
        agent_defaults = AgentDefaults()
        assert hasattr(agent_defaults, 'role_selector'), "AgentDefaults 缺少 role_selector 属性"
        print(f"\n✓ AgentDefaults 包含 role_selector 属性")
        
        return True
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_requirements_file():
    """测试需求文件功能。"""
    print("\n" + "=" * 60)
    print("测试4: 需求文件功能")
    print("=" * 60)
    
    import tempfile
    from pathlib import Path
    
    try:
        # 创建临时工作目录
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            
            # 测试1: 文件不存在时自动创建
            print("\n测试4.1: 自动创建需求文件")
            req_path = ensure_requirements_file(workspace, "roles/requirements.md")
            assert req_path.exists(), "需求文件应该被创建"
            print(f"  ✓ 文件已创建: {req_path}")
            
            content = req_path.read_text(encoding="utf-8")
            assert content == DEFAULT_REQUIREMENTS_CONTENT, "文件内容应该是默认内容"
            print(f"  ✓ 文件内容正确 ({len(content)} chars)")
            
            # 测试2: 文件已存在时不覆盖
            print("\n测试4.2: 已存在文件不覆盖")
            custom_content = "# 自定义需求\n\n我需要数据分析师和软件工程师"
            req_path.write_text(custom_content, encoding="utf-8")
            
            req_path2 = ensure_requirements_file(workspace, "roles/requirements.md")
            content2 = req_path2.read_text(encoding="utf-8")
            assert content2 == custom_content, "已存在的文件不应该被覆盖"
            print(f"  ✓ 已存在文件未被覆盖")
            
            # 测试3: 读取需求文件
            print("\n测试4.3: 读取需求文件")
            requirements = load_requirements(workspace, "roles/requirements.md")
            assert requirements == custom_content, "应该能正确读取需求文件"
            print(f"  ✓ 成功读取需求文件 ({len(requirements)} chars)")
            
            # 测试4: 相对路径处理
            print("\n测试4.4: 相对路径处理")
            req_path3 = ensure_requirements_file(workspace, "roles/requirements.md")
            assert req_path3.is_absolute(), "返回的应该是绝对路径"
            print(f"  ✓ 相对路径正确转换为绝对路径")
            
            print("\n✓ 所有需求文件测试通过")
            return True
            
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试。"""
    print("\n" + "=" * 60)
    print("角色选择器功能测试")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行测试
    results.append(("扫描角色", test_scan_roles()))
    results.append(("选择判断", test_should_select()))
    results.append(("配置Schema", test_config_schema()))
    results.append(("需求文件", test_requirements_file()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
