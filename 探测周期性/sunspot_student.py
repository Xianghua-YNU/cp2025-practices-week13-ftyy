#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
太阳黑子周期性分析 - 学生代码模板

请根据项目说明实现以下函数，完成太阳黑子效率与最优温度的计算。
"""

import numpy as np
import matplotlib.pyplot as plt

def load_sunspot_data(url):
    """
    从本地文件读取太阳黑子数据
    只保留第2(年份)和3(太阳黑子数)列
    """
    data = np.loadtxt(../susport_data.txt,usecols=(2, 3))
    years = data[:, 0]
    sunspots = data[:, 1]
    return years, sunspots

def plot_sunspot_data(years, sunspots):
    """
    绘制太阳黑子数据随时间变化图
    """
    plt.figure(figsize=(10, 4))
    plt.plot(years, sunspots, label='Sunspots')
    plt.xlabel('Year')
    plt.ylabel('Sunspot Number')
    plt.title('Sunspot Number vs Year')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_power_spectrum(sunspots):
    """
    计算太阳黑子数据的功率谱
    """
    n = len(sunspots)
    fft_result = np.fft.fft(sunspots - np.mean(sunspots))
    power = np.abs(fft_result)**2
    frequencies = np.fft.fftfreq(n, d=1)  # d=1表示采样间隔为1个月
    # 只保留正频率部分
    mask = frequencies > 0
    return frequencies[mask], power[mask]

def plot_power_spectrum(frequencies, power):
    """
    绘制功率谱图
    """
    plt.figure(figsize=(10, 4))
    plt.plot(frequencies, power)
    plt.xlabel('Frequency (1/month)')
    plt.ylabel('Power')
    plt.title('Power Spectrum of Sunspot Data')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def find_main_period(frequencies, power):
    """
    找出功率谱中的主周期
    """
    idx = np.argmax(power)
    main_frequency = frequencies[idx]
    main_period = 1 / main_frequency
    return main_period

def main():
    # 数据文件路径
    data = "sunspot_data.txt"
    
    # 1. 加载并可视化数据
    years, sunspots = load_sunspot_data(data)
    plot_sunspot_data(years, sunspots)
    
    # 2. 傅里叶变换分析
    frequencies, power = compute_power_spectrum(sunspots)
    plot_power_spectrum(frequencies, power)
    
    # 3. 确定主周期
    main_period = find_main_period(frequencies, power)
    print(f"\nMain period of sunspot cycle: {main_period:.2f} months")
    print(f"Approximately {main_period/12:.2f} years")

if __name__ == "__main__":
    main()
