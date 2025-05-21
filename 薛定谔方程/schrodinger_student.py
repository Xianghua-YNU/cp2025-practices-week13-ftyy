#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
薛定谔方程 - 方势阱能级计算

本模块实现了一维方势阱中粒子能级的计算方法。
"""

import numpy as np
import matplotlib.pyplot as plt

# 物理常数
HBAR = 1.0545718e-34  # 约化普朗克常数 (J·s)
ELECTRON_MASS = 9.1094e-31  # 电子质量 (kg)
EV_TO_JOULE = 1.6021766208e-19  # 电子伏转换为焦耳的系数


def calculate_y_values(E_values, V, w, m):
    """
    计算方势阱能级方程中的三个函数值
    
    参数:
        E_values (numpy.ndarray): 能量值数组 (eV)
        V (float): 势阱高度 (eV)
        w (float): 势阱宽度 (m)
        m (float): 粒子质量 (kg)
    
    返回:
        tuple: 包含三个numpy数组 (y1, y2, y3)，分别对应三个函数在给定能量值下的函数值
    """
    # TODO: 实现计算y1, y2, y3的代码 (约10行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 注意单位转换和避免数值计算中的溢出或下溢
    
    E_joules = E_values * EV_TO_JOULE
    V_joule = V * EV_TO_JOULE
    factor = (w ** 2 * m) / (2 * HBAR ** 2)
    sqrt_arg = factor * E_joules
    y1 = np.tan(np.sqrt(sqrt_arg))
    with np.errstate(divide='ignore', invalid='ignore'):
        y2 = np.sqrt((V_joule - E_joules) / E_joules)
        y3 = -np.sqrt(E_joules / (V_joule - E_joules))
    y1 = np.where(np.isfinite(y1), y1, np.nan)
    y2 = np.where(np.isfinite(y2), y2, np.nan)
    y3 = np.where(np.isfinite(y3), y3, np.nan)
    return y1, y2, y3


def plot_energy_functions(E_values, y1, y2, y3):
    """
    绘制能级方程的三个函数曲线
    
    参数:
        E_values (numpy.ndarray): 能量值数组 (eV)
        y1 (numpy.ndarray): 函数y1的值
        y2 (numpy.ndarray): 函数y2的值
        y3 (numpy.ndarray): 函数y3的值
    
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    # TODO: 实现绘制三个函数曲线的代码 (约15行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 使用不同颜色和线型，添加适当的标签、图例和标题
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(E_values, y1, 'b-', label=r'$y_1 = \tan\sqrt{w^2 m E / 2\hbar^2}$')
    ax.plot(E_values, y2, 'r--', label=r'$y_2 = \sqrt{(V-E)/E}$ (even parity)')
    ax.plot(E_values, y3, 'g-.', label=r'$y_3 = -\sqrt{E/(V-E)}$ (odd parity)')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlim(0, np.max(E_values))
    ax.set_ylim(-10, 10)
    ax.set_xlabel('Energy E (eV)')
    ax.set_ylabel('Function value')
    ax.set_title('Square Potential Well Energy Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def find_energy_level_bisection(n, V, w, m, precision=0.001, E_min=0.001, E_max=None):
    """
    使用二分法求解方势阱中的第n个能级
    
    参数:
        n (int): 能级序号 (0表示基态，1表示第一激发态，以此类推)
        V (float): 势阱高度 (eV)
        w (float): 势阱宽度 (m)
        m (float): 粒子质量 (kg)
        precision (float): 求解精度 (eV)
        E_min (float): 能量搜索下限 (eV)
        E_max (float): 能量搜索上限 (eV)，默认为V
    
    返回:
        float: 第n个能级的能量值 (eV)
    """
    # TODO: 实现二分法求解能级的代码 (约25行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 需要考虑能级的奇偶性，偶数能级使用偶宇称方程，奇数能级使用奇宇称方程
    
    if E_max is None:
        E_max = V - 1e-6  # 避免除零

    def even_func(E):
        E_joule = E * EV_TO_JOULE
        V_joule = V * EV_TO_JOULE
        factor = (w ** 2 * m) / (2 * HBAR ** 2)
        try:
            y1 = np.tan(np.sqrt(factor * E_joule))
            y2 = np.sqrt((V_joule - E_joule) / E_joule)
            return y1 - y2
        except Exception:
            return np.nan

    def odd_func(E):
        E_joule = E * EV_TO_JOULE
        V_joule = V * EV_TO_JOULE
        factor = (w ** 2 * m) / (2 * HBAR ** 2)
        try:
            y1 = np.tan(np.sqrt(factor * E_joule))
            y3 = -np.sqrt(E_joule / (V_joule - E_joule))
            return y1 - y3
        except Exception:
            return np.nan

    func = even_func if n % 2 == 0 else odd_func

    # 判断区间端点是否异号
    fa = func(E_min)
    fb = func(E_max)
    if np.isnan(fa) or np.isnan(fb):
        raise ValueError("区间端点函数值无效，请检查参数设置。")
    # 如果区间端点异号，直接用该区间
    if fa * fb < 0:
        a, b = E_min, E_max
    else:
        # 自动扫描，找到第n个根
        scan_points = 1000
        E_scan = np.linspace(E_min, E_max, scan_points)
        f_scan = func(E_scan)
        f_scan = np.where(np.isnan(f_scan), 1e6, f_scan)
        sign_changes = np.where(np.diff(np.sign(f_scan)))[0]
        if len(sign_changes) <= n:
            raise ValueError("指定区间内未找到足够的根，请检查参数设置。")
        a = E_scan[sign_changes[n]]
        b = E_scan[sign_changes[n] + 1]

    # 二分法
    while b - a > precision:
        mid = (a + b) / 2
        fm = func(mid)
        if np.isnan(fm):
            mid += precision * 0.1
            fm = func(mid)
        fa = func(a)
        if fa * fm < 0:
            b = mid
        else:
            a = mid
    energy_level = (a + b) / 2
    return energy_level


def main():
    """
    主函数，执行方势阱能级的计算和可视化
    """
    # 参数设置
    V = 20.0  # 势阱高度 (eV)
    w = 1e-9  # 势阱宽度 (m)
    m = ELECTRON_MASS  # 粒子质量 (kg)
    
    # 1. 计算并绘制函数曲线
    E_values = np.linspace(0.001, 19.999, 1000)  # 能量范围 (eV)
    y1, y2, y3 = calculate_y_values(E_values, V, w, m)
    fig = plot_energy_functions(E_values, y1, y2, y3)
    plt.savefig('energy_functions.png', dpi=300)
    plt.show()
    
    # 2. 使用二分法计算前6个能级
    energy_levels = []
    for n in range(6):
        energy = find_energy_level_bisection(n, V, w, m)
        energy_levels.append(energy)
        print(f"能级 {n}: {energy:.3f} eV")
    
    # 与参考值比较
    reference_levels = [0.318, 1.270, 2.851, 5.050, 7.850, 11.215]
    print("\n参考能级值:")
    for n, ref in enumerate(reference_levels):
        print(f"能级 {n}: {ref:.3f} eV")


if __name__ == "__main__":
    main()
