import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

def generate_all_figures(
    df_all: 'pd.DataFrame',
    model_names: list,
    model_shape_bias: dict,
    per_shape_bias: dict,
    per_texture_bias: dict,
    all_results: list,
    config: dict,
    categories_16: list,
):
    """
    Generate and save all five output figures based on analysis metrics.
    """
    figures_dir = config["figures_dir"]
    human_sb    = config["human_shape_bias"]
    published   = config.get("published_baselines", {})
    
    os.makedirs(figures_dir, exist_ok=True)

    vis_lookup = {
        (r['model'], r['file_name']): r['_vis']
        for r in all_results
        if '_vis' in r
    }

    print("  Generating Fig 1: Main shape bias comparison...")
    bar_labels, bar_values, bar_cis, bar_colors = [], [], [], []

    bar_labels.append("Human\n(Geirhos 2019)")
    bar_values.append(human_sb)
    bar_cis.append(None)
    bar_colors.append('#2e7d32')

    for name in model_names:
        sb, ci = model_shape_bias[name]
        bar_labels.append(f"{name.upper()}\n(our run)")
        bar_values.append(sb)
        bar_cis.append(ci)
        bar_colors.append('#1565c0')

    for name in model_names:
        pub = published.get(name)
        if pub is not None:
            bar_labels.append(f"{name.upper()}\n(Geirhos 2019)")
            bar_values.append(pub)
            bar_cis.append(None)
            bar_colors.append('#90a4ae')

    fig1, ax1 = plt.subplots(figsize=(max(8, len(bar_labels) * 1.8), 6))
    bars = ax1.bar(range(len(bar_labels)), bar_values,
                   color=bar_colors, alpha=0.85, edgecolor='white',
                   linewidth=1.5, width=0.6)

    for i, (val, ci) in enumerate(zip(bar_values, bar_cis)):
        if ci is not None:
            ax1.errorbar(i, val, yerr=[[val - ci[0]], [ci[1] - val]],
                         color='black', capsize=5, capthick=2, linewidth=2)

    for bar, val in zip(bars, bar_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

    ax1.axhline(0.5,    color='red',     linestyle='--', linewidth=1.5,
                alpha=0.7, label='Chance (0.50)')
    ax1.axhline(human_sb, color='#2e7d32', linestyle=':',  linewidth=1.5,
                alpha=0.5, label='Human baseline')

    ax1.set_xticks(range(len(bar_labels)))
    ax1.set_xticklabels(bar_labels, fontsize=10)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("Shape Bias  [0 = pure texture  →  1 = pure shape]", fontsize=11)
    ax1.set_title(
        "Shape Bias: Humans vs. CNNs on Cue-Conflict Images\n"
        "CNNs are strongly texture-biased; humans are strongly shape-biased",
        fontsize=12, fontweight='bold'
    )
    legend_handles = [
        mpatches.Patch(color='#2e7d32', label='Human participants'),
        mpatches.Patch(color='#1565c0', label='Our replicated results (±95% CI)'),
        mpatches.Patch(color='#90a4ae', label='Published results (Geirhos 2019)'),
    ]
    ax1.legend(handles=legend_handles, fontsize=9, loc='upper right')
    sns.despine()
    plt.tight_layout()
    path1 = f"{figures_dir}/fig1_main_shape_bias.png"
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"    Saved: {path1}")

    print("  Generating Fig 2: Per-category heatmap...")
    hm = pd.DataFrame(index=sorted(categories_16),
                      columns=model_names, dtype=float)
    for name in model_names:
        for cat in categories_16:
            hm.loc[cat, name] = per_shape_bias[name].get(cat, np.nan)

    hm['_avg'] = hm[model_names].mean(axis=1)
    hm = hm.sort_values('_avg', ascending=False).drop(columns=['_avg'])

    fig2, ax2 = plt.subplots(figsize=(max(5, len(model_names) * 2.8), 8))
    sns.heatmap(
        hm.astype(float), ax=ax2,
        annot=True, fmt='.2f', annot_kws={'size': 10},
        cmap='RdYlGn', vmin=0, vmax=1,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Shape Bias', 'shrink': 0.8}
    )
    ax2.set_title(
        "Per-Category Shape Bias\nGreen = CNN uses shape  |  Red = CNN uses texture",
        fontsize=12, fontweight='bold'
    )
    ax2.set_xlabel("Model", fontsize=11)
    ax2.set_ylabel("Shape Category (sorted by avg shape bias)", fontsize=11)
    ax2.tick_params(axis='x', labelsize=11)
    ax2.tick_params(axis='y', labelsize=10, rotation=0)
    plt.tight_layout()
    path2 = f"{figures_dir}/fig2_per_category_heatmap.png"
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"    Saved: {path2}")

    print("  Generating Fig 3: Confidence distributions...")
    n_m   = len(model_names)
    fig3, axes3 = plt.subplots(1, n_m, figsize=(7 * n_m, 5), sharey=True)
    if n_m == 1:
        axes3 = [axes3]
    fig3.suptitle(
        "Confidence: Shape-Category vs. Texture-Category Probabilities\n"
        "If texture confidence > shape confidence → model leans toward texture",
        fontsize=12, fontweight='bold'
    )
    for ax, name in zip(axes3, model_names):
        df_m = df_all[df_all['model'] == name]
        ax.hist(df_m['shape_confidence'],   bins=40, alpha=0.65, color='#1565c0',
                label=f"Shape conf  μ={df_m['shape_confidence'].mean():.3f}")
        ax.hist(df_m['texture_confidence'], bins=40, alpha=0.65, color='#c62828',
                label=f"Texture conf μ={df_m['texture_confidence'].mean():.3f}")
        ax.axvline(df_m['shape_confidence'].mean(),   color='#1565c0',
                   linewidth=2, linestyle='--')
        ax.axvline(df_m['texture_confidence'].mean(), color='#c62828',
                   linewidth=2, linestyle='--')
        ax.set_title(name.upper(), fontsize=12, fontweight='bold')
        ax.set_xlabel("Softmax Probability", fontsize=10)
        ax.set_ylabel("Number of Images", fontsize=10)
        ax.legend(fontsize=9)
        sns.despine(ax=ax)
    plt.tight_layout()
    path3 = f"{figures_dir}/fig3_confidence_distributions.png"
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"    Saved: {path3}")

    print("  Generating Fig 4: Example image grid...")
    primary = model_names[1]        # --------------------------------------------------------
    df_p    = df_all[df_all['model'] == primary]
    shape_ex   = df_p[df_p['decision'] == 'shape'].head(4)
    texture_ex = df_p[df_p['decision'] == 'texture'].head(4)
    n_ex = min(4, len(shape_ex), len(texture_ex))

    if n_ex > 0:
        fig4, axes4 = plt.subplots(2, n_ex, figsize=(n_ex * 3.5, 7))
        if n_ex == 1:
            axes4 = axes4.reshape(2, 1)
        fig4.suptitle(
            f"Cue-Conflict Images — {primary.upper()} Decisions\n"
            "Top: SHAPE decisions (green)  |  Bottom: TEXTURE decisions (red)",
            fontsize=11, fontweight='bold'
        )

        def _annotate(ax, row, decision_type):
            color = '#2e7d32' if decision_type == 'shape' else '#c62828'
            label = 'SHAPE ✓' if decision_type == 'shape' else 'TEXTURE ✗'
            vis   = vis_lookup.get((primary, row['file_name']))
            if vis is not None:
                ax.imshow(vis)
            ax.set_title(
                f"{label}\nShape: {row['shape_label']}  "
                f"Texture: {row['texture_label']}\n"
                f"Mapped: {str(row['top1_class_category'])} | Raw: {str(row['top1_class_name'])[:18]}",
                fontsize=7, color=color
            )
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

        for col, (_, row) in enumerate(shape_ex.head(n_ex).iterrows()):
            _annotate(axes4[0, col], row, 'shape')
        for col, (_, row) in enumerate(texture_ex.head(n_ex).iterrows()):
            _annotate(axes4[1, col], row, 'texture')

        plt.tight_layout()
        path4 = f"{figures_dir}/fig4_example_decisions.png"
        fig4.savefig(path4, dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print(f"    Saved: {path4}")
    else:
        print("    Skipped Fig 4 (not enough examples)")

    print("  Generating Fig 5: Decision breakdown stacked bars...")
    fig5, axes5 = plt.subplots(1, n_m, figsize=(10 * n_m, 6), sharey=True)
    if n_m == 1:
        axes5 = [axes5]
    fig5.suptitle(
        "Decision Breakdown per Shape Category\n"
        "Blue = Shape  |  Red = Texture  |  Grey = Neither (excluded from shape bias)",
        fontsize=12, fontweight='bold'
    )

    cats_sorted = sorted(categories_16)
    x = np.arange(len(cats_sorted))

    for ax, name in zip(axes5, model_names):
        df_m = df_all[df_all['model'] == name]
        shape_f, texture_f, neither_f = [], [], []
        for cat in cats_sorted:
            sub = df_m[df_m['shape_label'] == cat]
            n = len(sub)
            if n == 0:
                shape_f.append(0); texture_f.append(0); neither_f.append(0)
            else:
                shape_f.append((sub['decision'] == 'shape').sum() / n)
                texture_f.append((sub['decision'] == 'texture').sum() / n)
                neither_f.append((sub['decision'] == 'neither').sum() / n)

        ax.bar(x, shape_f,   color='#1565c0', alpha=0.85, label='Shape')
        ax.bar(x, texture_f, bottom=shape_f,  color='#c62828', alpha=0.85, label='Texture')
        ax.bar(x, neither_f,
               bottom=[s + t for s, t in zip(shape_f, texture_f)],
               color='#90a4ae', alpha=0.85, label='Neither')
        ax.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(cats_sorted, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of decisions", fontsize=10)
        ax.set_title(name.upper(), fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        sns.despine(ax=ax)

    plt.tight_layout()
    path5 = f"{figures_dir}/fig5_decision_breakdown.png"
    fig5.savefig(path5, dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print(f"    Saved: {path5}")

    print("\n  All figures saved.")