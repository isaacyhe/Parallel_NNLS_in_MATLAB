import numpy as np
import pandas as pd
import argparse
from scipy.ndimage import zoom

def load_grid_data(filename):
    """Load CSV data and reshape into a 2D grid."""
    df = pd.read_csv(filename)
    
    # Round coordinates to avoid floating-point precision issues
    df["x"] = df["x"].round(5)
    df["y"] = df["y"].round(5)
    
    # Pivot into a 2D grid and handle NaNs
    grid = df.pivot(index="x", columns="y", values="z").values
    x_coords = df["x"].unique()
    y_coords = df["y"].unique()
    
    # Ensure numeric type and handle NaNs
    grid = grid.astype(np.float64)
    grid = np.nan_to_num(grid, nan=np.nan)  # Replace NaNs with NaN, or use another value
    
    return grid, x_coords, y_coords

def scale_grid(grid, target_dim=None, scale_factor=None):
    """Scale a 2D grid to target size or by a factor."""
    if target_dim:
        # Calculate scale factors for both axes
        scale_x = target_dim / grid.shape[0]
        scale_y = target_dim / grid.shape[1]
        scale_factor = (scale_x, scale_y)
    elif scale_factor:
        scale_factor = (scale_factor, scale_factor)
    else:
        raise ValueError("Specify either `target_dim` or `scale_factor`.")
    
    # Check if scaling factors are reasonable (e.g., greater than 0)
    if any(s <= 0 for s in scale_factor):
        raise ValueError("Scaling factors must be positive.")
    
    # Apply scaling with linear interpolation
    scaled_grid = zoom(grid, scale_factor, order=1)
    return scaled_grid

def generate_scaled_coordinates(original_coords, target_dim):
    """Generate new coordinates for the scaled grid."""
    # Generate new coordinates based on target_dim
    new_coords = np.linspace(original_coords.min(), original_coords.max(), target_dim)
    return new_coords

def save_grid(grid, original_x_coords, original_y_coords, target_dim, output_file):
    """Save the scaled grid back into CSV with the same format as the original."""
    # Generate new coordinates for the scaled grid
    new_x_coords = generate_scaled_coordinates(original_x_coords, target_dim)
    new_y_coords = generate_scaled_coordinates(original_y_coords, target_dim)
    
    # Ensure grid matches new coordinates in dimensions
    if grid.shape[0] != target_dim or grid.shape[1] != target_dim:
        raise ValueError(f"Scaled grid dimensions {grid.shape} do not match target dimensions ({target_dim}, {target_dim}).")
    
    # Create a DataFrame from the scaled grid
    scaled_df = pd.DataFrame(grid, index=new_x_coords, columns=new_y_coords)

    # Add 'x' and 'y' columns for melting
    scaled_df['x'] = scaled_df.index
    scaled_df = scaled_df.reset_index(drop=True).melt(id_vars=["x"], var_name="y", value_name="z")
    
    # Save to CSV
    scaled_df.to_csv(output_file, index=False)

def main():
    parser = argparse.ArgumentParser(description="Scale a 2D grid from CSV.")
    parser.add_argument("input", help="Input CSV file (x,y,z columns)")
    parser.add_argument("output", help="Output CSV file")
    parser.add_argument("--target_dim", type=int, help="Target grid dimension (e.g., 80)")
    parser.add_argument("--scale_factor", type=float, help="Scaling factor (e.g., 2.0)")
    
    args = parser.parse_args()
    
    # Load and validate input
    grid, x_coords, y_coords = load_grid_data(args.input)
    print(f"Original grid shape: {grid.shape}")
    
    # Scale the grid
    scaled_grid = scale_grid(
        grid, 
        target_dim=args.target_dim, 
        scale_factor=args.scale_factor
    )
    print(f"Scaled grid shape: {scaled_grid.shape}")
    
    # Save output in the same format as the original
    save_grid(scaled_grid, x_coords, y_coords, args.target_dim, args.output)
    print(f"Saved scaled grid to: {args.output}")

if __name__ == "__main__":
    main()