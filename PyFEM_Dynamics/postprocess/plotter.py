import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import List, Optional, Tuple
from core.node import Node
from core.element import TrussElement2D

class Plotter:
    """
    负责生成数据可视化图表。
    """
    
    @staticmethod
    def setup_paper_style():
        """
        初始化全局绘图样式。
        要求：Times New Roman 字体，清晰的字号，允许 LaTeX 公式渲染，及高分辨率设定。
        """
        plt.style.use('default')
        mpl.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'mathtext.fontset': 'stix',
            'axes.linewidth': 1.0,
            'grid.alpha': 0.5,
            'grid.linestyle': '--'
        })
        
    @staticmethod
    def plot_structure(nodes: List[Node], 
                       elements: List[TrussElement2D], 
                       U: Optional[np.ndarray] = None, 
                       scale_factor: float = 1.0, 
                       element_colors: Optional[np.ndarray] = None,
                       title: str = "Structure Deformation",
                       save_path: Optional[str] = None):
        """
        绘制结构未变形与变形对比图。
        :param nodes: 节点列表
        :param elements: 单元列表
        :param U: 节点全局位移向量，如果提供，将画出变形后结构
        :param scale_factor: 位移放大系数，以使其在图中可视
        :param element_colors: 每个单元的值（如轴力、损伤指标等），用于给变形单元染色
        :param title: 图像标题
        :param save_path: 文件保存路径 (含扩展名)
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 1. 绘制初始(未变形)结构，使用浅灰虚线
        for el in elements:
            x_coords = [el.node1.x, el.node2.x]
            y_coords = [el.node1.y, el.node2.y]
            ax.plot(x_coords, y_coords, 'k--', linewidth=1.0, alpha=0.4, 
                    label="Undeformed" if el.element_id == 0 else "")
            
        # 2. 绘制变形后结构
        if U is not None:
            # 建立 colormap
            cmap = plt.cm.coolwarm
            norm = None
            if element_colors is not None:
                norm = mpl.colors.Normalize(vmin=np.min(element_colors), vmax=np.max(element_colors))
            
            for i, el in enumerate(elements):
                # 提取节点变形后的坐标
                u1 = U[el.node1.dofs[0]]
                v1 = U[el.node1.dofs[1]]
                u2 = U[el.node2.dofs[0]]
                v2 = U[el.node2.dofs[1]]
                
                x_def = [el.node1.x + u1 * scale_factor, el.node2.x + u2 * scale_factor]
                y_def = [el.node1.y + v1 * scale_factor, el.node2.y + v2 * scale_factor]
                
                color = 'b' if element_colors is None else cmap(norm(element_colors[i]))
                ax.plot(x_def, y_def, '-', color=color, linewidth=2.0)
                
            # 如果提供了颜色映射，则添加 colorbar
            if element_colors is not None and norm is not None:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Element Value', rotation=270, labelpad=15)
                
            # 绘制变形后的节点
            node_x_def = [n.x + U[n.dofs[0]] * scale_factor for n in nodes]
            node_y_def = [n.y + U[n.dofs[1]] * scale_factor for n in nodes]
            ax.scatter(node_x_def, node_y_def, c='r', s=20, zorder=5, label="Nodes (Deformed)")
            
        # 美化图形
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_title(title)
        
        # 消除重复图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            
        ax.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"结构变形图已保存至: {save_path}")
        else:
            plt.show()
            
        plt.close(fig)

    @staticmethod
    def plot_time_history(t: np.ndarray, 
                          data: List[np.ndarray], 
                          labels: List[str], 
                          ylabel: str = "Acceleration ($m/s^2$)",
                          title: str = "Transient Response Time History",
                          save_path: Optional[str] = None):
        """
        绘制时间响应曲线。
        :param t: 时间步数组 ( shape: (N,) )
        :param data: 需绘制的数据序列列表，每个数据的 shape 应为 (N,)
        :param labels: 数据对应的曲线图例名列表
        :param ylabel: Y 轴名称 (支持 LaTeX)
        :param title: 图像标题
        :param save_path: 文件保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        lines = ['-', '--', '-.', ':']
        
        for i, (series, label) in enumerate(zip(data, labels)):
            ax.plot(t, series, color=colors[i % len(colors)], 
                    linestyle=lines[i % len(lines)], linewidth=1.5, label=label)
            
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend(loc='best', framealpha=0.9, edgecolor='black')
        
        # x轴范围限制
        ax.set_xlim(t[0], t[-1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"时程图已保存至: {save_path}")
        else:
            plt.show()
            
        plt.close(fig)

    @staticmethod
    def plot_ai4m_sample(X: np.ndarray, Y: np.ndarray, sample_idx: int, t_array: np.ndarray, save_path: Optional[str] = None):
        """
        绘制数据集中某一个特定样本的时程以及对应的损伤信息柱状组合图。
        :param X: 形状为 ( channels, timesteps ) 的序列数据
        :param Y: 形状为 ( num_elements, ) 的标签数据 (损伤系数)
        :param sample_idx: 用于标注图像标题的具体样本编号
        :param t_array: 时间轴
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1]})
        
        # Left: Time History of sensors
        ax_left = axs[0]
        for c in range(X.shape[0]):
            ax_left.plot(t_array, X[c, :], label=f"Sensor {c+1}", linewidth=1.0)
        ax_left.set_xlabel("Time (s)")
        ax_left.set_ylabel("Acceleration ($m/s^2$)")
        ax_left.set_title(f"Sample {sample_idx} - Sensor Signals")
        ax_left.grid(True)
        ax_left.legend()
        ax_left.set_xlim(t_array[0], t_array[-1])
        
        # Right: Damage state of elements
        ax_right = axs[1]
        elements = np.arange(len(Y)) + 1
        bars = ax_right.bar(elements, Y * 100, color='skyblue', edgecolor='black')
        
        # Color specific bars that are damaged (< 100%)
        for i, bar in enumerate(bars):
            if Y[i] < 0.99:  # Considered damaged
                bar.set_color('salmon')
                bar.set_edgecolor('black')
                
        ax_right.set_ylim(0, 110)
        ax_right.axhline(100, color='gray', linestyle='--', linewidth=1)
        ax_right.set_xlabel("Element ID")
        ax_right.set_ylabel("Stiffness Ratio E/E0 (%)")
        ax_right.set_title("Element Damage State")
        ax_right.set_xticks(elements)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"AI4M数据样本图已保存至: {save_path}")
        else:
            plt.show()
            
        plt.close(fig)

    @staticmethod
    def plot_truss_vm_frame(
        nodes: List[Node],
        elements: List[TrussElement2D],
        vm_values: np.ndarray,
        title: str,
        save_path: str,
        U: Optional[np.ndarray] = None,
        scale_factor: float = 1.0,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ):
        """
        绘制桁架单元 von Mises 应力(近似)云图单帧，支持位移叠加。
        """
        if len(elements) != len(vm_values):
            raise ValueError("vm_values length must match number of elements")

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 1. 绘制初始参考结构 (浅灰色)
        for element in elements:
            ax.plot([element.node1.x, element.node2.x], 
                    [element.node1.y, element.node2.y], 
                    "k--", linewidth=0.8, alpha=0.2)

        cmap = plt.cm.viridis
        vmin = np.min(vm_values) if vmin is None else vmin
        vmax = np.max(vm_values) if vmax is None else vmax
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        # 2. 绘制应力单元 (应用位移)
        for idx, element in enumerate(elements):
            n1, n2 = element.node1, element.node2
            
            x1, y1 = n1.x, n1.y
            x2, y2 = n2.x, n2.y
            
            if U is not None:
                x1 += U[n1.dofs[0]] * scale_factor
                y1 += U[n1.dofs[1]] * scale_factor
                x2 += U[n2.dofs[0]] * scale_factor
                y2 += U[n2.dofs[1]] * scale_factor
                
            color = cmap(norm(vm_values[idx]))
            ax.plot([x1, x2], [y1, y2], "-", color=color, linewidth=2.5, zorder=4)

        # 绘制节点
        if U is not None:
            node_x = [n.x + U[n.dofs[0]] * scale_factor for n in nodes]
            node_y = [n.y + U[n.dofs[1]] * scale_factor for n in nodes]
        else:
            node_x = [n.x for n in nodes]
            node_y = [n.y for n in nodes]
            
        ax.scatter(node_x, node_y, s=15, c="black", zorder=5)
        
        ax.set_aspect("equal", "box")
        ax.set_xlabel("X Coordinate (m)")
        ax.set_ylabel("Y Coordinate (m)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, linestyle="--")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("von Mises Stress (Pa)", rotation=270, labelpad=16)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
