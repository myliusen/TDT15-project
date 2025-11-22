import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def plot_yield_loadings(loadings: np.ndarray, name=None) -> None:
    """Plot loading vectors as line plots for yield curve factors"""
    maturities = np.arange(1, loadings.shape[1] + 1)
    dimension = loadings.shape[0]
    
    plt.figure(figsize=(10, 6))
    for i in range(dimension):
        plt.plot(maturities, loadings[i, :], label=f'Component {i+1}')
    
    plt.xlabel('Maturity')
    plt.ylabel('Component loadings')
    plt.title(f'Loadings for Yield Curve: {name}' if name else 'Loadings for Yield Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_marginal_r_squares(fred_md: pd.DataFrame, scores: pd.DataFrame, 
                             figsize=(16, 12)) -> None:
    """Plot marginal R-squares of each variable in fred_md against the first three factors."""
    categories = fred_md.attrs['categories']
    unique_categories = sorted(set(categories.values()))
    
    # Define colors for each category
    category_colors = {
        cat: color for cat, color in zip(
            unique_categories,
            plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
        )
    }
    
    # Create sorted column order by category
    sorted_cols = sorted(fred_md.columns, key=lambda x: (categories[x], x))
    
    # Calculate marginal R-squares for each PC
    all_marginal_r_squares = {f'C{i+1}': [] for i in range(3)}
    colors_sorted = []
    
    for col in sorted_cols:
        colors_sorted.append(category_colors[categories[col]])
        
        for i in range(3):
            # Get the PCA factor
            pca_factor = scores[f'C{i+1}']
            
            # Get the variable data (align indices)
            common_idx = fred_md.index.intersection(scores.index)
            X = fred_md.loc[common_idx, col].values.reshape(-1, 1)
            y = pca_factor.loc[common_idx].values
            
            # Fit simple linear regression
            reg = LinearRegression()
            reg.fit(X, y)
            
            # Predict and calculate R-square
            y_pred = reg.predict(X)
            r2 = r2_score(y, y_pred)
            all_marginal_r_squares[f'C{i+1}'].append(max(0, r2))
    
    # Create 3-row plot
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    x_pos = np.arange(len(sorted_cols))
    
    for idx, pc in enumerate(['C1', 'C2', 'C3']):
        ax = axes[idx]
        
        # Plot bars with category colors
        ax.bar(x_pos, all_marginal_r_squares[pc], color=colors_sorted, 
               edgecolor='black', linewidth=0.3)
        
        # Formatting
        ax.set_xlabel('Variable Index', fontsize=11)
        ax.set_ylabel('R-square', fontsize=11)
        ax.set_title(f'Marginal R-squares for {pc}', fontsize=13)
        ax.set_xticks(x_pos[::4])
        ax.set_xticklabels(x_pos[::4] + 1)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(max(all_marginal_r_squares[pc]), 0.1) * 1.1)
    
    # Add legend for categories
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=category_colors[cat], 
                                     edgecolor='black', label=cat) 
                       for cat in unique_categories]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99),
               ncol=2, fontsize=9, title='Categories', title_fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    
    plt.show()

def plot_keras_training_loss(history, title='Training Loss Over Time'):
    """Plot the training loss curve from a Keras training history object."""
    plt.figure(figsize=(5, 3))
    plt.plot(history.history['loss'], linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_latent_space_trajectory_3d_static(scores_df, title="Latent Space Trajectory"):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create colormap based on date
    dates = scores_df.index
    date_nums = (dates - dates.min()).days.values
    
    # Plot trajectory as a line
    scatter = ax.scatter(
        scores_df['C1'],
        scores_df['C2'],
        scores_df['C3'],
        c=date_nums,
        cmap='viridis',
        s=20,
        alpha=0.6,
        edgecolors='none'
    )
    
    # Add connecting lines
    ax.plot(
        scores_df['C1'].values,
        scores_df['C2'].values,
        scores_df['C3'].values,
        color='gray',
        alpha=0.3,
        linewidth=0.5
    )
    
    # Labels and title
    ax.set_xlabel('C1')
    ax.set_ylabel('C2')
    ax.set_zlabel('C3')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Time', fontsize=11)
    cbar.set_ticks([date_nums.min(), date_nums.max()])
    cbar.set_ticklabels([dates.min().strftime('%Y'), dates.max().strftime('%Y')])
        
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig