import os
import glob
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import Affine

# -------------------- CSV Processing Function --------------------
def process_csv(csv_folder, report_path):
    """
    Read all CSV files in the folder.
    - Compute per-file statistics (rows, cols, mean, std, min, max).
    - Check column consistency across files (raise error if mismatched).
    - Concatenate all valid files into one DataFrame.
    - Compute overall mean and final shape.
    - Write statistics to report_path.
    - Save the concatenated matrix to 'merged_data.csv'.
    """
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    if not csv_files:
        print("Warning: No CSV files found.")
        return

    print(f"[Self-check] Found {len(csv_files)} CSV files.")

    stats = []
    dfs = []
    ref_columns = None
    ref_cols_count = None

    for file in csv_files:
        df = pd.read_csv(file)
        rows, cols = df.shape

        # Calculate statistics on numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            mean_val = std_val = min_val = max_val = np.nan
        else:
            mean_val = numeric_df.mean().mean()
            std_val = numeric_df.std().mean()
            min_val = numeric_df.min().min()
            max_val = numeric_df.max().max()

        stats.append({
            'filename': os.path.basename(file),
            'rows': rows,
            'cols': cols,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val
        })

        # Check column consistency with the first file
        if ref_columns is None:
            ref_columns = df.columns.tolist()
            ref_cols_count = cols
        else:
            if cols != ref_cols_count or not df.columns.equals(pd.Index(ref_columns)):
                error_msg = f"File '{os.path.basename(file)}' cannot be merged: column count or column names differ from the first file."
                print(error_msg)  # Console log
                raise ValueError(error_msg)

        dfs.append(df)

    # Merge all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    total_rows, total_cols = merged_df.shape

    numeric_merged = merged_df.select_dtypes(include=[np.number])
    overall_mean = numeric_merged.mean().mean() if not numeric_merged.empty else np.nan

    # Write statistics report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("===== CSV Statistical Report =====\n\n")
        f.write("Per-file statistics:\n")
        for s in stats:
            f.write(f"Filename: {s['filename']}\n")
            f.write(f"  Rows: {s['rows']}\n")
            f.write(f"  Columns: {s['cols']}\n")
            f.write(f"  Mean: {s['mean']:.6f}\n" if not np.isnan(s['mean']) else "  Mean: NaN\n")
            f.write(f"  Std Dev: {s['std']:.6f}\n" if not np.isnan(s['std']) else "  Std Dev: NaN\n")
            f.write(f"  Min: {s['min']:.6f}\n" if not np.isnan(s['min']) else "  Min: NaN\n")
            f.write(f"  Max: {s['max']:.6f}\n" if not np.isnan(s['max']) else "  Max: NaN\n")
            f.write("\n")

        f.write("===== Merged Dataset Overview =====\n")
        f.write(f"Final matrix shape (rows, columns): ({total_rows}, {total_cols})\n")
        f.write(f"Overall mean (all numeric values): {overall_mean:.6f}\n" if not np.isnan(overall_mean) else "Overall mean: NaN\n")

    # Save the merged matrix as a new CSV file (Extra task)
    merged_csv_path = "merged_data.csv"
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"Success: Merged matrix saved to '{merged_csv_path}'.")
    print(f"CSV statistical report generated: {report_path}")

# -------------------- Raster Processing Function --------------------
def process_rs(rs_folder, report_path):
    """
    Read all .tif files in the folder.
    For each file, report the number of bands.
    For each band, compute and report: min, max, mean, std.
    Write results to report_path.
    """
    tif_files = glob.glob(os.path.join(rs_folder, "*.tif"))
    if not tif_files:
        print("Warning: No TIFF files found.")
        return

    print(f"[Self-check] Found {len(tif_files)} TIFF files.")

    with open(report_path, 'w', encoding='utf-8') as f:
        for tif in tif_files:
            with rasterio.open(tif) as src:
                band_count = src.count
                print(f"[Self-check] Processing '{os.path.basename(tif)}' -> {band_count} bands detected.")

                f.write(f"===== {os.path.basename(tif)} =====\n")
                f.write(f"Number of Bands: {band_count}\n\n")

                for band_idx in range(1, band_count + 1):
                    band_data = src.read(band_idx)
                    band_min = band_data.min()
                    band_max = band_data.max()
                    band_mean = band_data.mean()
                    band_std = band_data.std()

                    # Use description if available, otherwise generic "Band n"
                    band_name = src.descriptions[band_idx - 1] if src.descriptions and src.descriptions[band_idx - 1] else f"Band {band_idx}"

                    f.write(f"Band: {band_name}\n")
                    f.write(f"  Min: {band_min:.6f}\n")
                    f.write(f"  Max: {band_max:.6f}\n")
                    f.write(f"  Mean: {band_mean:.6f}\n")
                    f.write(f"  Std Dev: {band_std:.6f}\n\n")

                f.write("\n")  # Separate files

    print(f"Raster statistical report generated: {report_path}")

# -------------------- Transformation & Save Function (Extra Training) --------------------
def transform_and_save_rs(rs_folder):
    """
    Perform geometric transformations (rotate 90°, flip left-right, crop center 100x100)
    on all .tif files in rs_folder and save the results to a new folder 'rs_transformed'.
    Preserves ALL bands in the output files.
    """
    tif_files = glob.glob(os.path.join(rs_folder, "*.tif"))
    if not tif_files:
        print("Warning: No TIFF files found for transformation.")
        return

    # Create output folder if it doesn't exist
    output_folder = "rs_transformed"
    os.makedirs(output_folder, exist_ok=True)

    print("\n--- Starting Extra Training: Transform and Save Images ---")

    for tif in tif_files:
        with rasterio.open(tif) as src:
            data = src.read()  # Reads all bands -> shape (bands, height, width)
            base_name = os.path.splitext(os.path.basename(tif))[0]
            print(f"[Self-check] Transforming '{base_name}' with {src.count} bands...")

            # 1. Rotate 90 degrees counter-clockwise
            rotated = np.rot90(data, k=1, axes=(1, 2))
            out_path_rot = os.path.join(output_folder, f"{base_name}_rot90.tif")
            profile = src.profile
            profile.update(height=rotated.shape[1], width=rotated.shape[2])
            with rasterio.open(out_path_rot, 'w', **profile) as dst:
                dst.write(rotated)
            print(f"  Saved rotated image: {out_path_rot}")

            # 2. Flip Left-Right (Horizontal Mirror)
            flipped_lr = data[:, :, ::-1]
            out_path_lr = os.path.join(output_folder, f"{base_name}_flipLR.tif")
            profile_flip = src.profile
            with rasterio.open(out_path_lr, 'w', **profile_flip) as dst:
                dst.write(flipped_lr)
            print(f"  Saved left-right flipped image: {out_path_lr}")

            # 3. Crop Center 100x100 (if image is larger than 100)
            h, w = data.shape[1], data.shape[2]
            if h > 100 and w > 100:
                y_start = (h - 100) // 2
                x_start = (w - 100) // 2
                cropped = data[:, y_start:y_start+100, x_start:x_start+100]

                new_transform = src.transform * Affine.translation(x_start, y_start)
                out_path_crop = os.path.join(output_folder, f"{base_name}_crop100.tif")
                profile_crop = src.profile
                profile_crop.update(height=100, width=100, transform=new_transform)

                with rasterio.open(out_path_crop, 'w', **profile_crop) as dst:
                    dst.write(cropped)
                print(f"  Saved center-cropped image (100x100): {out_path_crop}")
            else:
                print(f"  Skipped cropping for '{base_name}': image size ({h}x{w}) is smaller than 100x100.")

# -------------------- Main Function --------------------
def main():
    csv_folder = "csv_data"
    rs_folder = "rs_data"
    csv_report = "csv_report.txt"
    rs_report = "rs_report.txt"

    # Self-check: Verify folders exist
    if not os.path.exists(csv_folder):
        print(f"Warning: Folder '{csv_folder}' not found.")
    if not os.path.exists(rs_folder):
        print(f"Warning: Folder '{rs_folder}' not found.")

    # 1. Process CSV files
    print("\n--- Processing CSV Data ---")
    try:
        process_csv(csv_folder, csv_report)
    except Exception as e:
        print(f"CSV processing terminated: {e}")

    # 2. Process Raster files (Statistics)
    print("\n--- Processing Raster Data (Statistics) ---")
    process_rs(rs_folder, rs_report)

    # 3. Extra Training: Transform and save images
    transform_and_save_rs(rs_folder)

    print("\n--- All tasks completed successfully. ---")

if __name__ == "__main__":
    main()