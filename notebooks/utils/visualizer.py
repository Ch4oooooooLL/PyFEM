import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import warnings

plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_structure_topology(nodes, elements, save_path: Optional[str] = None,
                           show_node_ids: bool = True, show_elem_ids: bool = True,
                           show_constraints: bool = True, constraints: Optional[List] = None):
    """绘制结构拓扑图
    
    Args:
        nodes: 节点列表或节点字典
        elements: 单元列表或单元字典
        save_path: 保存路径
        show_node_ids: 是否显示节点编号
        show_elem_ids: 是否显示单元编号
        show_constraints: 是否显示约束
        constraints: 约束信息列表
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if isinstance(nodes, dict):
        node_list = [nodes[i] for i in sorted(nodes.keys())]
    else:
        node_list = nodes
    
    if isinstance(elements, dict):
        elem_list = [elements[i] for i in sorted(elements.keys())]
    else:
        elem_list = elements
    
    x_coords = [n['coords'][0] for n in node_list]
    y_coords = [n['coords'][1] for n in node_list]
    
    for elem in elem_list:
        if isinstance(elem, dict):
            node_ids = elem['nodes']
        else:
            node_ids = [elem.node1.node_id, elem.node2.node_id]
        
        n1 = node_list[node_ids[0]]
        n2 = node_list[node_ids[1]]
        
        x = [n1['coords'][0], n2['coords'][0]]
        y = [n1['coords'][1], n2['coords'][1]]
        ax.plot(x, y, 'b-', linewidth=2, zorder=1)
        
        if show_elem_ids:
            mid_x = (x[0] + x[1]) / 2
            mid_y = (y[0] + y[1]) / 2
            if isinstance(elem, dict):
                elem_id = elem['id']
            else:
                elem_id = elem.element_id
            ax.text(mid_x, mid_y, str(elem_id), fontsize=9, ha='center', va='bottom',
                   color='red', fontweight='bold', zorder=2)
    
    ax.scatter(x_coords, y_coords, c='blue', s=100, zorder=3)
    
    if show_node_ids:
        for n in node_list:
            ax.text(n['coords'][0], n['coords'][1], str(n['id']), 
                   fontsize=10, ha='right', va='top', 
                   color='white', fontweight='bold', zorder=4)
    
    if show_constraints and constraints:
        for bc in constraints:
            node_id = bc['node_id']
            node = node_list[node_id]
            x, y = node['coords'][0], node['coords'][1]
            
            if 'ux' in bc['constraints'] and 'uy' in bc['constraints']:
                ax.plot(x, y, 's', markersize=15, color='gray', zorder=2)
            elif 'uy' in bc['constraints']:
                ax.plot(x, y, '^', markersize=12, color='gray', zorder=2)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Structure Topology', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_deformation(nodes, elements, U, scale_factor: float = 1.0,
                    colors: Optional[np.ndarray] = None, save_path: Optional[str] = None,
                    show_node_ids: bool = True):
    """绘制结构变形图
    
    Args:
        nodes: 节点列表或节点字典
        elements: 单元列表或单元字典
        U: 位移向量
        scale_factor: 位移放大系数
        colors: 单元颜色数组（可选）
        save_path: 保存路径
        show_node_ids: 是否显示节点编号
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if isinstance(nodes, dict):
        node_list = [nodes[i] for i in sorted(nodes.keys())]
    else:
        node_list = nodes
    
    if isinstance(elements, dict):
        elem_list = [elements[i] for i in sorted(elements.keys())]
    else:
        elem_list = elements
    
    num_nodes = len(node_list)
    num_dofs = len(U)
    dofs_per_node = num_dofs // num_nodes if num_nodes > 0 else 2
    
    orig_x = np.array([n['coords'][0] for n in node_list])
    orig_y = np.array([n['coords'][1] for n in node_list])
    
    if len(U) >= num_nodes * 2:
        disp_x = U[0::dofs_per_node][:num_nodes] * scale_factor
        disp_y = U[1::dofs_per_node][:num_nodes] * scale_factor
    else:
        disp_x = np.zeros(num_nodes)
        disp_y = np.zeros(num_nodes)
    
    deformed_x = orig_x + disp_x
    deformed_y = orig_y + disp_y
    
    if colors is None:
        colors = np.zeros(len(elem_list))
    
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=np.min(colors), vmax=np.max(colors))
    
    for i, elem in enumerate(elem_list):
        if isinstance(elem, dict):
            node_ids = elem['nodes']
        else:
            node_ids = [elem.node1.node_id, elem.node2.node_id]
        
        x_orig = [orig_x[node_ids[0]], orig_x[node_ids[1]]]
        y_orig = [orig_y[node_ids[0]], orig_y[node_ids[1]]]
        ax.plot(x_orig, y_orig, 'b--', linewidth=1, alpha=0.5, zorder=1)
        
        x_def = [deformed_x[node_ids[0]], deformed_x[node_ids[1]]]
        y_def = [deformed_y[node_ids[0]], deformed_y[node_ids[1]]]
        
        color = cmap(norm(colors[i])) if i < len(colors) else 'blue'
        ax.plot(x_def, y_def, '-', linewidth=2.5, color=color, zorder=2)
    
    ax.scatter(orig_x, orig_y, c='blue', s=50, alpha=0.5, label='Original', zorder=1)
    ax.scatter(deformed_x, deformed_y, c='red', s=80, label='Deformed', zorder=3)
    
    if show_node_ids:
        for i, n in enumerate(node_list):
            ax.text(deformed_x[i], deformed_y[i], str(n['id']),
                   fontsize=9, ha='right', va='bottom', color='red', zorder=4)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'Structure Deformation (Scale Factor: {scale_factor})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Value', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_time_history(t, data_list, labels, ylabel, title, save_path: Optional[str] = None,
                     figsize: Tuple = (10, 6)):
    """绘制时程曲线
    
    Args:
        t: 时间数组
        data_list: 数据列表
        labels: 标签列表
        ylabel: Y轴标签
        title: 图表标题
        save_path: 保存路径
        figsize: 图形大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for data, label in zip(data_list, labels):
        ax.plot(t, data, label=label, linewidth=1.5)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_damage_comparison(y_true, y_pred, element_ids=None, title: str = 'Damage Prediction Comparison',
                          save_path: Optional[str] = None, figsize: Tuple = (12, 5)):
    """绘制损伤预测对比图
    
    Args:
        y_true: 真实损伤值
        y_pred: 预测损伤值
        element_ids: 单元ID列表
        title: 图表标题
        save_path: 保存路径
        figsize: 图形大小
    """
    if element_ids is None:
        element_ids = range(len(y_true))
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    x = np.arange(len(element_ids))
    
    axes[0].bar(x - 0.2, y_true, 0.4, label='True', alpha=0.8)
    axes[0].bar(x + 0.2, y_pred, 0.4, label='Predicted', alpha=0.8)
    axes[0].set_xlabel('Element ID', fontsize=12)
    axes[0].set_ylabel('Damage Coefficient', fontsize=12)
    axes[0].set_title('Damage Comparison', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(i) for i in element_ids])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    im = axes[1].imshow(np.column_stack([y_true, y_pred]), aspect='auto', cmap='coolwarm')
    axes[1].set_xlabel('Type', fontsize=12)
    axes[1].set_ylabel('Element ID', fontsize=12)
    axes[1].set_title('Damage Heatmap', fontsize=12, fontweight='bold')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['True', 'Predicted'])
    plt.colorbar(im, ax=axes[1])
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metrics_scatter(y_true, y_pred, model_name: str = 'Model', 
                        save_path: Optional[str] = None, figsize: Tuple = (8, 8)):
    """绘制预测 vs 真实散点图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        model_name: 模型名称
        save_path: 保存路径
        figsize: 图形大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidths=0.5)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    ax.text(0.05, 0.95, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}',
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('True Value', fontsize=12)
    ax.set_ylabel('Predicted Value', fontsize=12)
    ax.set_title(f'{model_name}: Predicted vs True', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_error_histogram(errors, bins: int = 30, title: str = 'Error Distribution',
                        save_path: Optional[str] = None, figsize: Tuple = (8, 6)):
    """绘制误差分布直方图
    
    Args:
        errors: 误差数组
        bins: 直方图bin数量
        title: 图表标题
        save_path: 保存路径
        figsize: 图形大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(errors, bins=bins, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(x=np.mean(errors), color='g', linestyle='-', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    
    ax.set_xlabel('Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_calibration_curve(y_true, y_pred, n_bins: int = 10, title: str = 'Calibration Curve',
                          save_path: Optional[str] = None, figsize: Tuple = (8, 6)):
    """绘制校准曲线
    
    Args:
        y_true: 真实标签
        y_pred: 预测概率
        n_bins: bin数量
        title: 图表标题
        save_path: 保存路径
        figsize: 图形大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    true_probs = []
    pred_probs = []
    
    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if np.sum(mask) > 0:
            true_probs.append(np.mean(y_true[mask]))
            pred_probs.append(np.mean(y_pred[mask]))
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax.plot(pred_probs, true_probs, 'o-', label='Model', markersize=6)
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None,
                          figsize: Tuple = (12, 5)):
    """绘制训练历史曲线
    
    Args:
        history: 训练历史字典
        save_path: 保存路径
        figsize: 图形大小
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Loss History', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    if 'val_mae' in history:
        axes[1].plot(epochs, history['val_mae'], 'g-', label='Val MAE')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('MAE History', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_stress_distribution(nodes, elements, stress_values, title: str = 'Stress Distribution',
                             save_path: Optional[str] = None, figsize: Tuple = (12, 6)):
    """绘制应力分布图
    
    Args:
        nodes: 节点列表
        elements: 单元列表
        stress_values: 应力值数组
        title: 图表标题
        save_path: 保存路径
        figsize: 图形大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(nodes, dict):
        node_list = [nodes[i] for i in sorted(nodes.keys())]
    else:
        node_list = nodes
    
    if isinstance(elements, dict):
        elem_list = [elements[i] for i in sorted(elements.keys())]
    else:
        elem_list = elements
    
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=np.min(stress_values), vmax=np.max(stress_values))
    
    for i, elem in enumerate(elem_list):
        if isinstance(elem, dict):
            node_ids = elem['nodes']
        else:
            node_ids = [elem.node1.node_id, elem.node2.node_id]
        
        n1 = node_list[node_ids[0]]
        n2 = node_list[node_ids[1]]
        
        x = [n1['coords'][0], n2['coords'][0]]
        y = [n1['coords'][1], n2['coords'][1]]
        
        color = cmap(norm(stress_values[i]))
        ax.plot(x, y, '-', linewidth=4, color=color, zorder=1)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Stress (Pa)', fontsize=12)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
