#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一讲梯度下降法 - 快速演示脚本
Quick Demo Script for Lesson 1: Gradient Descent

一键运行所有演示，适合快速体验和学习
"""

import sys
import os
import subprocess
import time

def print_header(title):
    """打印标题"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def print_section(title):
    """打印章节标题"""
    print(f"\n📚 {title}")
    print("-" * 40)

def run_with_delay(func, delay=3):
    """运行函数并等待"""
    func()
    print(f"\n⏳ 等待 {delay} 秒后继续...")
    time.sleep(delay)

def demo_simple_concepts():
    """演示简单概念"""
    print_section("1. 简单概念演示")
    print("运行简化的梯度下降演示...")
    
    try:
        from simple_gradient_descent_demo import simple_gradient_descent_demo, linear_regression_step_by_step
        
        print("\n🔍 一维函数优化演示：")
        simple_gradient_descent_demo()
        
        print("\n🔍 线性回归逐步演示：")
        linear_regression_step_by_step()
        
        print("✅ 简单概念演示完成！")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保所有依赖都已正确安装")

def demo_complete_training():
    """演示完整训练过程"""
    print_section("2. 完整训练过程演示")
    print("运行完整的线性回归训练...")
    
    try:
        from linear_regression_gradient_descent import main
        main()
        print("✅ 完整训练演示完成！")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保所有依赖都已正确安装")

def demo_data_tools():
    """演示数据工具"""
    print_section("3. 数据生成工具演示")
    print("运行数据生成和处理工具...")
    
    try:
        from data_utils import demo_data_generation
        demo_data_generation()
        print("✅ 数据工具演示完成！")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保所有依赖都已正确安装")

def check_dependencies():
    """检查依赖"""
    print_section("依赖检查")
    
    required_packages = ['numpy', 'matplotlib', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: 已安装")
        except ImportError:
            print(f"❌ {package}: 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装：")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ 所有依赖都已正确安装！")
    return True

def show_menu():
    """显示菜单"""
    print("\n" + "="*60)
    print("🎯 李宏毅机器学习2021 - 第一讲：梯度下降法")
    print("="*60)
    print("\n请选择要运行的演示：")
    print("1. 简单概念演示（推荐新手）")
    print("2. 完整训练过程演示")
    print("3. 数据生成工具演示")
    print("4. 运行所有演示")
    print("5. 检查依赖")
    print("0. 退出")
    print("\n" + "-"*60)

def interactive_mode():
    """交互模式"""
    while True:
        show_menu()
        
        try:
            choice = input("\n请输入选择 (0-5): ").strip()
            
            if choice == '0':
                print("\n👋 感谢使用！继续学习机器学习！")
                break
                
            elif choice == '1':
                demo_simple_concepts()
                
            elif choice == '2':
                demo_complete_training()
                
            elif choice == '3':
                demo_data_tools()
                
            elif choice == '4':
                print_header("运行所有演示")
                run_with_delay(demo_simple_concepts, 2)
                run_with_delay(demo_complete_training, 2)
                run_with_delay(demo_data_tools, 2)
                print("\n🎉 所有演示完成！")
                
            elif choice == '5':
                check_dependencies()
                
            else:
                print("❌ 无效选择，请输入 0-5 之间的数字")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")

def auto_mode():
    """自动模式 - 运行所有演示"""
    print_header("自动演示模式")
    print("将依次运行所有演示，每个演示之间会有短暂停顿")
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请先安装缺少的包")
        return
    
    # 运行所有演示
    run_with_delay(demo_simple_concepts, 3)
    run_with_delay(demo_complete_training, 3)
    run_with_delay(demo_data_tools, 3)
    
    print_header("所有演示完成")
    print("🎉 恭喜！你已经完成了第一讲的所有演示")
    print("\n💡 接下来你可以：")
    print("   1. 尝试修改参数，观察不同的效果")
    print("   2. 阅读代码，理解实现细节")
    print("   3. 完成README中的思考题和扩展练习")
    print("   4. 继续学习后续课程内容")

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == '--auto':
            auto_mode()
        elif sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("使用方法:")
            print("  python run_demo.py          # 交互模式")
            print("  python run_demo.py --auto   # 自动运行所有演示")
            print("  python run_demo.py --help   # 显示帮助信息")
        else:
            print(f"❌ 未知参数: {sys.argv[1]}")
            print("使用 --help 查看帮助信息")
    else:
        interactive_mode()

if __name__ == "__main__":
    main()
