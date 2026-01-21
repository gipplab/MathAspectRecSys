import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast

def create_joyplot(input_file, output_image):
    name_mapping = {
        "goldRiM_cos": "GoldRiM Rec.",
        "goldRiM_top": "GoldRiM Global",
        "gldRimcit_cos": "GoldRiM Citations",
        "pwc_recmnd": "PwC Rec.",
        "pwc_topdoc": "PwC Global"
    }

    # Define Colors (Pastel shades)
    # Shades of Blue for GoldRiM, Shades of Orange for PwC
    blues = ["#BCA0FF", "#9FB9FF", "#A3E2FF"] # Light/Pastel Blues
    oranges = ["#FFC977", "#FFA7A7"]           # Light/Pastel Oranges
    
    # 1. Parse the data file
    data_list = []
    try:
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    # Split line into key and list part
                    name, list_str = line.split('=', 1)
                    name = name.strip()

                    # Apply mapping if it exists, otherwise use original name
                    display_name = name_mapping.get(name, name)

                    # Convert string list to actual Python list
                    values = ast.literal_eval(list_str.strip())
                    for v in values:
                        data_list.append({'Category': display_name, 'Value': v})
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    df = pd.DataFrame(data_list)

    # 2. Setup aesthetic parameters
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    # Sort categories by mean value for a cleaner visual flow
    category_order = df.groupby('Category')['Value'].mean().sort_values().index
    
    # Generate a beautiful pastel palette
    # Using 'husl' or 'pastel' with a slight fade effect
    # pal = sns.color_palette("pastel", n_colors=len(category_order))
    # 3. Create Custom Palette based on sorted order
    palette_dict = {}
    b_idx, o_idx = 0, 0
    for cat in category_order:
        if cat.startswith("G"):
            palette_dict[cat] = blues[b_idx % len(blues)]
            b_idx += 1
        elif cat.startswith("P"):
            palette_dict[cat] = oranges[o_idx % len(oranges)]
            o_idx += 1
        else:
            palette_dict[cat] = "#D3D3D3"

    # print(category_order)
    # print(custom_pal)

    # 3. Initialize the FacetGrid
    # hspace=-0.5 creates the "overlapping" joyplot effect
    g = sns.FacetGrid(df, row="Category", hue="Category", aspect=8, height=0.8, 
                      palette=palette_dict, row_order=category_order)

    # 4. Draw the densities
    # 'fill=True' and 'alpha' create the fading/soft look
    g.map(sns.kdeplot, "Value", bw_adjust=.8, clip=(0.0,1.0), fill=True, alpha=0.6, linewidth=0,zorder=1)
    g.map(sns.kdeplot, "Value", clip=(0.0,1.0), color="white", lw=2, bw_adjust=.8,zorder=2)

    # Add a baseline for each distribution
    g.refline(y=-0.14, linewidth=1.5, linestyle="-", color=None, clip_on=False, alpha=0.6,zorder=3)

    # 5. Add custom labels inside each plot
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .15, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes, fontsize=14)

    g.map(label, "Value")

    # 6. Final visual adjustments
    g.figure.subplots_adjust(hspace=-0.6) # Overlap distributions
    g.set_titles("")
    g.set(yticks=[], ylabel="", xlabel="", xlim=(0.0, 1))
    g.despine(bottom=True, left=True)
    
    # Optional: Title
    g.figure.suptitle('Cosine Similarity of Qwen3-7B Embeddings', fontsize=16, y=0.98, x=0.58)

    # Save the result
    plt.savefig(output_image, bbox_inches='tight', dpi=300)
    print(f"Joyplot saved as {output_image}")

if __name__ == "__main__":
    # Ensure your data.txt is in the same directory
    create_joyplot('data_new.txt', 'joyplot_new.png') # data_new.txt is provided below

# data_new.txt
# clusters = {
#     "Specialization / Restriction": {
#         "size": 37.83,
#         "labels": [
#             ("revised", 14), ("described", 13), ("investigated", 11),
#             ("considered", 10), ("applied", 9), ("introduced", 8),
#             ("methods", 7), ("methodology", 7), ("approach", 6),
#             ("question", 5), ("problem", 4), ("related", 3),
#             ("preliminary", 2), ("reviewed", 1),
#         ],
#     },
#     "Generalization / Modification": {
#         "size": 29.30,
#         "labels": [
#             ("generalization", 22), ("generalized", 18), ("extension", 14),
#             ("modification", 12), ("variant", 10), ("analogue", 8),
#             ("characterization", 6), ("extends concept", 4),
#             ("generalized version", 3), ("proved generalizations", 3),
#         ],
#     },
#     "Results": {
#         "size": 21.04,
#         "labels": [
#             ("result", 24), ("achieved", 18), ("obtained", 15),
#             ("determined", 12), ("observed", 10), ("conclusion given", 8),
#             ("main result", 6), ("precise result", 4), ("result implies", 3),
#         ],
#     },
#     "Prove / Cases": {
#         "size": 11.83,
#         "labels": [
#             ("proof", 22), ("proved", 18), ("theorem proven", 15),
#             ("provided proof", 12), ("proved cases", 10),
#             ("complete proof", 8), ("claims proved", 6),
#             ("proof involves", 4), ("re-proved result", 3),
#             ("verify cases", 2),
#         ],
#     },
# }
