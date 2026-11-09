import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import seaborn as sns
import argparse
import os
import sys

mpl.rcParams['pdf.fonttype'] = 42

# One color per GIN cluster (index = cluster number).
myclustercolors = [
    '#D865A6','#7C2880','#E23A36','#E23A36','#E23A36','#000000',
    '#1F8241','#E23A36','#E23A36','#000000','#189BD7','#1F8241',
    '#000000','#E23A36','#189BD7','#1F8241','#1F8241','#189BD7',
    '#000000','#E23A36','#8C8C8C','#7C2880','#D865A6','#000000',
    '#E23A36','#189BD7','#189BD7','#7C2880','#F58220','#189BD7'
]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot protein domain diagrams from a long-format CSV.")
    parser.add_argument('-i', '--input', required=True, help="Path to the input CSV (e.g., final_long_table.csv)")
    parser.add_argument('-o', '--output', default='protein_domains_diagram.pdf', help="Name of the output image file")

    # Mutually exclusive protein selection options
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        '-p', '--protein',
        help="UniprotID of a single protein to plot (e.g., P12345)"
    )
    selection.add_argument(
        '-l', '--protein-list',
        dest='protein_list',
        help="Path to a plain-text file with one UniprotID per line"
    )

    return parser.parse_args()


def load_protein_ids(path):
    """Read a newline-separated file of UniprotIDs, ignoring blank lines and comments."""
    if not os.path.isfile(path):
        sys.exit(f"Error: protein list file not found: {path}")
    with open(path) as fh:
        ids = [line.strip() for line in fh if line.strip() and not line.startswith('#')]
    if not ids:
        sys.exit(f"Error: no UniprotIDs found in {path}")
    return ids


def add_position_labels(ax, start, end, y_pos, domain_height, fontsize=6):
    """
    Draw tick marks and numeric labels for the start and end of a domain/block.
    Ticks drop below the rectangle; labels sit just below the ticks.
    """
    tick_len   = domain_height * 0.55   # length of the small vertical tick
    label_pad  = tick_len + 0.04        # gap between bottom edge and label

    bottom = y_pos - domain_height / 2  # bottom edge of the rectangle

    for pos, ha in [(start, 'center'), (end, 'center')]:
        # Vertical tick line
        ax.plot(
            [pos, pos],
            [bottom, bottom - tick_len],
            color='#444444', linewidth=0.7, zorder=5, clip_on=True
        )
        # Numeric label
        ax.text(
            pos, bottom - label_pad, str(int(pos)),
            ha=ha, va='top', fontsize=fontsize,
            color='#444444', zorder=5, clip_on=True
        )


def select_proteins(df, requested=None):
    """Keep the rows of the requested UniprotIDs, in the requested order.
    With requested=None every protein in df is kept, in file order.
    Returns (filtered df, ordered list of UniprotIDs).
    """
    if requested is None:
        return df, list(df['UniprotID'].unique())

    missing = [uid for uid in requested if uid not in df['UniprotID'].values]
    if missing:
        print(f"Warning: the following UniprotIDs were not found in the data and will be skipped: {missing}")
    proteins = [uid for uid in requested if uid in df['UniprotID'].values]
    if not proteins:
        raise ValueError("None of the requested UniprotIDs were found in the input data.")
    df = df[df['UniprotID'].isin(proteins)].copy()
    df['_order'] = df['UniprotID'].map({uid: i for i, uid in enumerate(proteins)})
    df.sort_values('_order', inplace=True, kind='stable')
    df.drop(columns='_order', inplace=True)
    return df, proteins


def plot_protein_diagrams(df, proteins=None, track_height=2.5, width=12):
    """Draw one track per protein from a long-format domain table
    (columns: UniprotID, Gene name, Organism, Domain type, Domain name,
    Start, End; Domain type is one of 'Total protein', 'Domain', 'IDR',
    'IDR block'). Returns the matplotlib Figure.
    """
    df, proteins = select_proteins(df, proteins)

    # axes_style instead of set_theme keeps the style local to this figure.
    with sns.axes_style("ticks"):
        # Extra vertical room per protein to accommodate the position labels below domains
        fig, ax = plt.subplots(figsize=(width, track_height * len(proteins)))

        y_ticks  = []
        y_labels = []
        y_pos    = 0
        y_step   = 2   # distance between each protein track
        max_x    = 0   # to scale the X-axis later

        for uid in proteins:
            prot_data = df[df['UniprotID'] == uid]

            # Grab metadata for the Y-axis label
            first_row = prot_data.iloc[0]
            gene = first_row.get('Gene name')
            org  = first_row.get('Organism')
            gene = gene if pd.notna(gene) else "Unknown Gene"
            org  = org  if pd.notna(org)  else "Unknown Org"

            label = f"{gene}\n{org}\n{uid}"
            y_ticks.append(y_pos)
            y_labels.append(label)

            # Sizing parameters
            base_height   = 0.2
            domain_height = 0.4
            idr_height    = 0.1

            for _, row in prot_data.iterrows():
                dtype = row['Domain type']
                start = row['Start']
                end   = row['End']
                width_ = end - start
                dname = row['Domain name']

                if end > max_x:
                    max_x = end

                if dtype == 'Total protein':
                    rect = patches.Rectangle(
                        (start, y_pos - base_height / 2), width_, base_height,
                        facecolor='#808080', edgecolor='black', zorder=1
                    )
                    ax.add_patch(rect)
                    add_position_labels(ax, start, end, y_pos, base_height)

                elif dtype == 'Domain':
                    rect = patches.Rectangle(
                        (start, y_pos - domain_height / 2), width_, domain_height,
                        facecolor='yellow', edgecolor='black', zorder=2
                    )
                    ax.add_patch(rect)

                    center_x = start + width_ / 2
                    ax.text(
                        center_x, y_pos, str(dname),
                        ha='center', va='center', fontsize=8,
                        color='black', zorder=4, clip_on=True
                    )
                    add_position_labels(ax, start, end, y_pos, domain_height)

                elif dtype == 'IDR block':
                    try:
                        cluster_idx = int(float(dname))
                        color = myclustercolors[cluster_idx % len(myclustercolors)]
                    except (ValueError, TypeError):
                        color = '#CCCCCC'

                    rect = patches.Rectangle(
                        (start, y_pos - domain_height / 2), width_, domain_height,
                        facecolor=color, edgecolor='black', zorder=2
                    )
                    ax.add_patch(rect)

                    center_x = start + width_ / 2
                    ax.text(
                        center_x, y_pos, str(dname),
                        ha='center', va='center', fontsize=8,
                        color='white', zorder=4, clip_on=True
                    )
                    add_position_labels(ax, start, end, y_pos, domain_height)

                # The IDR is a thin blue bar floating above the protein track.
                elif dtype == 'IDR':
                    y_offset_above = y_pos + domain_height / 2 + 0.1
                    rect = patches.Rectangle(
                        (start, y_offset_above), width_, idr_height,
                        facecolor='blue', edgecolor='none', zorder=3
                    )
                    ax.add_patch(rect)
                    add_position_labels(ax, start, end, y_offset_above, idr_height, fontsize=5)

            y_pos += y_step

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Amino acid position", fontsize=12, fontweight='bold')

        ax.set_xlim(0, max_x * 1.05)
        ax.set_ylim(-y_step, y_pos)

        sns.despine(ax=ax, left=True)
        ax.tick_params(axis='y', length=0)

        fig.tight_layout()
    return fig


def main():
    args = parse_arguments()

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    if args.protein:
        requested = [args.protein]
    elif args.protein_list:
        requested = load_protein_ids(args.protein_list)
    else:
        requested = None   # plot everything

    try:
        fig = plot_protein_diagrams(df, requested)
    except ValueError as err:
        sys.exit(f"Error: {err}")

    fig.savefig(args.output)
    print(f"Plot generated successfully as '{args.output}'")


if __name__ == "__main__":
    main()
