"""
Bump Chart Generator for Domain Flow Analysis (4 Layers)

This module creates bump charts showing the flow of domains through four layers:
1. Search Query (manual_label_domain_counts)
2. Web Search API (search_total_counts)
3. GPT-5 Final Response (citations_total_counts)
4. Domain Trustworthiness Summary (domain_trustworthiness_summary)

Author: Data Science Assistant
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set style
plt.style.use("default")


class BumpChartGenerator4Layer:
    """Generate bump charts for domain flow analysis across four layers."""

    def __init__(self, data_dir: str, output_dir: str = None):
        """
        Initialize the bump chart generator.

        Args:
            data_dir (str): Directory containing the CSV data files
            output_dir (str): Directory to save output plots (defaults to data_dir/plots)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "plots"
        self.output_dir.mkdir(exist_ok=True)

    def load_region_data(self, region: str) -> tuple:
        """
        Load data for a specific region across all four layers.

        Args:
            region (str): Region code ('UK', 'US', 'EU'/'Europe')

        Returns:
            tuple: (layer1_data, layer2_data, layer3_data, layer4_data)
        """
        # Handle region naming inconsistency
        region_layer1 = region
        region_layer2 = region
        region_layer3 = region
        region_layer4 = region

        if region == "EU":
            region_layer2 = "Europe"
            region_layer3 = "Europe"
            region_layer4 = "Europe"

        # Load layer 1: Manual label domain counts
        layer1_file = self.data_dir / f"manual_label_domain_counts_{region_layer1}.csv"
        layer1_data = pd.read_csv(layer1_file)

        # Load layer 2: Search total counts
        layer2_file = self.data_dir / f"search_total_counts_{region_layer2}.csv"
        layer2_data = pd.read_csv(layer2_file)

        # Load layer 3: Citations total counts
        layer3_file = self.data_dir / f"citations_total_counts_{region_layer3}.csv"
        layer3_data = pd.read_csv(layer3_file)

        # Load layer 4: Domain trustworthiness summary
        layer4_file = self.data_dir / "domain_trustworthiness_summary.csv"
        layer4_data = pd.read_csv(layer4_file)
        # Filter for the specific region
        layer4_data = layer4_data[layer4_data["region"] == region_layer4].copy()

        return layer1_data, layer2_data, layer3_data, layer4_data

    def get_unique_domains_and_colors(
        self,
        layer1: pd.DataFrame,
        layer2: pd.DataFrame,
        layer3: pd.DataFrame,
        layer4: pd.DataFrame,
    ) -> dict:
        """
        Get all unique domains from the top N domains across all four layers WITHIN A SINGLE REGION
        and assign colors to them. This creates a region-specific color mapping - each region gets
        its own independent color palette.

        Args:
            layer1 (pd.DataFrame): Top N Search Query domains for this region (already filtered)
            layer2 (pd.DataFrame): Top N Web Search API domains for this region (already filtered)
            layer3 (pd.DataFrame): Top N GPT-5 Final Response domains for this region (already filtered)
            layer4 (pd.DataFrame): Top N Domain Trustworthiness domains for this region (already filtered)

        Returns:
            dict: Mapping of domain names to colors for this specific region
        """
        # Get all unique domains across all layers
        all_domains = set()
        all_domains.update(layer1["common_domain"].unique())
        all_domains.update(layer2["common_domain"].unique())
        all_domains.update(layer3["common_domain"].unique())
        all_domains.update(layer4["common_domain"].unique())

        # Sort domains for consistent color assignment
        all_domains = sorted(list(all_domains))

        # Generate enough colors using multiple palettes
        color_palettes = [
            sns.color_palette("Set2", 8),
            sns.color_palette("tab20b", 20),
        ]
        all_colors = []
        for palette in color_palettes:
            all_colors.extend(palette)

        # Assign colors to domains
        domain_colors = {}
        for i, domain in enumerate(all_domains):
            domain_colors[domain] = all_colors[i % len(all_colors)]

        return domain_colors

    def process_layer_data(
        self, data: pd.DataFrame, count_col: str, top_n: int = 10
    ) -> pd.DataFrame:
        """
        Process layer data to get top domains and their rankings.

        Args:
            data (pd.DataFrame): Raw data for the layer
            count_col (str): Name of the count column ('frequency' or 'total_count' or 'trustworthy_count')
            top_n (int): Number of top domains to include

        Returns:
            pd.DataFrame: Processed data with rankings
        """
        # Sort by count and get top N
        processed = data.sort_values(count_col, ascending=False).head(top_n).copy()

        # Add ranking (1-based)
        processed["rank"] = range(1, len(processed) + 1)

        return processed[["common_domain", count_col, "rank"]]

    def create_flow_data(
        self,
        layer1: pd.DataFrame,
        layer2: pd.DataFrame,
        layer3: pd.DataFrame,
        layer4: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create flow data showing how domains move between layers.

        Args:
            layer1, layer2, layer3, layer4 (pd.DataFrame): Processed layer data

        Returns:
            pd.DataFrame: Flow data with domain positions across layers
        """
        # Get all unique domains across layers
        all_domains = set()
        all_domains.update(layer1["common_domain"])
        all_domains.update(layer2["common_domain"])
        all_domains.update(layer3["common_domain"])
        all_domains.update(layer4["common_domain"])

        flow_data = []

        for domain in all_domains:
            # Get positions in each layer (None if not present)
            l1_rank = (
                layer1[layer1["common_domain"] == domain]["rank"].iloc[0]
                if domain in layer1["common_domain"].values
                else None
            )
            l2_rank = (
                layer2[layer2["common_domain"] == domain]["rank"].iloc[0]
                if domain in layer2["common_domain"].values
                else None
            )
            l3_rank = (
                layer3[layer3["common_domain"] == domain]["rank"].iloc[0]
                if domain in layer3["common_domain"].values
                else None
            )
            l4_rank = (
                layer4[layer4["common_domain"] == domain]["rank"].iloc[0]
                if domain in layer4["common_domain"].values
                else None
            )

            # Get counts for bubble sizes
            # Layer 1 uses 'frequency' column
            l1_count = (
                layer1[layer1["common_domain"] == domain]["frequency"].iloc[0]
                if domain in layer1["common_domain"].values
                else 0
            )
            # Layers 2, 3, 4 use 'total_count' column
            l2_count = (
                layer2[layer2["common_domain"] == domain]["total_count"].iloc[0]
                if domain in layer2["common_domain"].values
                else 0
            )
            l3_count = (
                layer3[layer3["common_domain"] == domain]["total_count"].iloc[0]
                if domain in layer3["common_domain"].values
                else 0
            )
            l4_count = (
                layer4[layer4["common_domain"] == domain]["trustworthy_count"].iloc[0]
                if domain in layer4["common_domain"].values
                else 0
            )

            flow_data.append(
                {
                    "domain": domain,
                    "layer1_rank": l1_rank,
                    "layer2_rank": l2_rank,
                    "layer3_rank": l3_rank,
                    "layer4_rank": l4_rank,
                    "layer1_count": l1_count,
                    "layer2_count": l2_count,
                    "layer3_count": l3_count,
                    "layer4_count": l4_count,
                }
            )

        return pd.DataFrame(flow_data)

    def plot_bump_chart(
        self,
        flow_data: pd.DataFrame,
        region: str,
        domain_colors: dict,
        figsize: tuple = (26, 20),
    ):
        """
        Create the bump chart visualization.

        Args:
            flow_data (pd.DataFrame): Flow data for the region
            region (str): Region name for the title
            domain_colors (dict): Mapping of domain names to colors
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Layer positions (4 layers now)
        layer_positions = [1, 2, 3, 4]
        layer_names = [
            "Search Query",
            "Web Search API",
            "GPT-5 Final Response",
            "Trustworthy",
        ]

        # Filter to only show domains that appear in at least one layer's top 10
        visible_domains = flow_data[
            (flow_data["layer1_rank"].notna())
            | (flow_data["layer2_rank"].notna())
            | (flow_data["layer3_rank"].notna())
            | (flow_data["layer4_rank"].notna())
        ].copy()

        # Plot flow lines and bubbles
        for _, row in visible_domains.iterrows():
            domain = row["domain"]
            # Use consistent color from domain colors for this region
            color = domain_colors.get(domain, "gray")

            # Collect valid positions and ranks
            positions = []
            ranks = []
            counts = []

            for layer_idx, (rank_col, count_col) in enumerate(
                [
                    ("layer1_rank", "layer1_count"),
                    ("layer2_rank", "layer2_count"),
                    ("layer3_rank", "layer3_count"),
                    ("layer4_rank", "layer4_count"),
                ]
            ):
                if pd.notna(row[rank_col]):
                    positions.append(layer_positions[layer_idx])
                    ranks.append(row[rank_col])
                    counts.append(row[count_col])

            # Draw flow line if domain appears in multiple layers
            if len(positions) > 1:
                ax.plot(
                    positions, ranks, color=color, linewidth=20, alpha=0.3, zorder=1
                )

            # Draw bubbles for each layer where domain appears
            for pos, rank, count in zip(positions, ranks, counts):
                # Bubble radius
                bubble_radius = 0.25

                # Draw bubble
                circle = Circle(
                    (pos, rank), radius=bubble_radius, color=color, alpha=1, zorder=3
                )
                ax.add_patch(circle)

                # Add count text inside bubble
                ax.text(
                    pos,
                    rank,
                    str(int(count)),
                    ha="center",
                    va="center",
                    fontsize=28,
                    fontweight="bold",
                    color="white",
                    zorder=4,
                )

                # Add domain name below the bubble, centered, horizontal
                ax.text(
                    pos,
                    rank + 0.32,  # Position below bubble
                    domain,
                    ha="center",
                    va="top",
                    fontsize=30,
                    fontweight="normal",
                    color="black",
                    zorder=4,
                    rotation=0,  # Horizontal, not rotated
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.8,
                        edgecolor="none",
                    ),
                )

        # Customize the plot with more space for domain labels
        ax.set_xlim(0.5, 4.5)  # Extended to 4.5 for 4 layers
        ax.set_ylim(-0.2, 11.8)  # Extended range for more vertical spacing
        ax.invert_yaxis()  # Rank 1 at top

        # Set layer labels
        ax.set_xticks(layer_positions)
        ax.set_xticklabels(layer_names, fontsize=30, fontweight="bold")
        ax.tick_params(axis="x", pad=-100)

        # Set rank labels
        ax.set_yticks(range(1, 11))
        ax.set_yticklabels([f"{i}" for i in range(1, 11)], fontsize=36)

        # Remove spines and grid
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Add region label with more spacing from plot
        ax.text(
            0.02,
            0.98,
            f"{region}",
            transform=ax.transAxes,
            fontsize=50,
            fontweight="bold",
            va="top",
            ha="left",
        )

        plt.tight_layout(pad=0, rect=[0, 0.0, 1, 1])
        return fig

    def generate_chart_for_region(
        self, region_code: str, region_display: str, top_n: int = 10
    ):
        """
        Generate bump chart for a single region.

        Args:
            region_code (str): Region code for file loading
            region_display (str): Region name for display
            top_n (int): Number of top domains to include in each layer
        """
        print(f"Generating 4-layer bump chart for {region_display}...")

        try:
            # Load data
            layer1, layer2, layer3, layer4 = self.load_region_data(region_code)

            # Process each layer to get top N domains first
            layer1_processed = self.process_layer_data(layer1, "frequency", top_n)
            layer2_processed = self.process_layer_data(layer2, "total_count", top_n)
            layer3_processed = self.process_layer_data(layer3, "total_count", top_n)
            layer4_processed = self.process_layer_data(layer4, "trustworthy_count", top_n)

            # Get unique domains and assign colors AFTER filtering to top N
            domain_colors = self.get_unique_domains_and_colors(
                layer1_processed, layer2_processed, layer3_processed, layer4_processed
            )

            # Create flow data
            flow_data = self.create_flow_data(
                layer1_processed, layer2_processed, layer3_processed, layer4_processed
            )

            # Generate plot
            fig = self.plot_bump_chart(flow_data, region_display, domain_colors)

            # Save plot (use "europe" filename for EU region)
            filename_region = (
                "europe" if region_display == "EU" else region_display.lower()
            )
            output_file = self.output_dir / f"bump_chart_4layer_{filename_region}.png"
            fig.savefig(
                output_file,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.1,
                facecolor="white",
            )
            plt.close(fig)

            print(f"✓ Saved: {output_file}")

        except Exception as e:
            print(f"✗ Error generating chart for {region_display}: {str(e)}")
            import traceback

            traceback.print_exc()

    def generate_all_charts(self, top_n: int = 10):
        """
        Generate bump charts for all three regions separately.

        Args:
            top_n (int): Number of top domains to include in each layer
        """
        regions = [
            ("UK", "UK"),
            ("US", "US"),
            ("EU", "EU"),  # Display as EU instead of Europe
        ]

        for region_code, region_display in regions:
            # Generate each region separately
            self.generate_chart_for_region(region_code, region_display, top_n)


def main():
    """Main function to generate all 4-layer bump charts."""

    # Set up paths
    data_dir = "1_data_analysis/results"
    output_dir = "1_data_analysis/plots"

    # Create generator and generate all charts
    generator = BumpChartGenerator4Layer(data_dir, output_dir)
    generator.generate_all_charts(top_n=10)

    print("\nAll 4-layer bump charts generated successfully!")


if __name__ == "__main__":
    main()
