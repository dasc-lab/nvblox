import os
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint

def load_esdf_with_distance(file_path):
    """
    Load an ESDF .ply file and extract points (x, y, z) and intensity (distance).
    """

    with open(file_path, "r") as f:
        lines = f.readlines()

    header_ended = False
    data_start_idx = 0
    for i, line in enumerate(lines):
        if "end_header" in line:
            header_ended = True
            data_start_idx = i + 1
            break

    if not header_ended:
        raise ValueError("The input file does not appear to be a valid .ply file.")

 
    # Load the point data (x, y, z, intensity)
    data = np.loadtxt(lines[data_start_idx:], dtype=float)
    if data.shape[1] != 4:
        raise ValueError("The input .ply file must have 4 columns: x, y, z, intensity.")

    # Total number of points
    total_points = data[:, :3]
    distances= data[:, 3]

    return total_points, distances 
        

def process_violation_ply_files(environment, ref_filenames, voxel_size=0.02):

    environment_results = {} 

    subfolders = [f for f in os.listdir(environment)]

    subfolders.sort()

    # For each odometry covariance error
    for subfolder in subfolders:

        dir_path = os.path.join(environment, subfolder)

        # Get all .ply files in the current subfolder
        ply_files = [f for f in os.listdir(dir_path) if f.endswith('.ply')]
        esdf_files = [f for f in os.listdir(dir_path) if f.endswith('esdf.ply')]

        if not ply_files:
            logging.warning(f"No .ply files found in folder {dir_path}.")
            return

        folder_results = {} 

        # Logging for process the files (replace with your desired logic)
        logging.info(f"Processing folder: {dir_path}")

        for i, method in enumerate(ref_filenames, start=1):
            ply_file = method + ".ply"
            esdf_file = method + "_esdf.ply"

            file_path = os.path.join(dir_path, ply_file)
            esdf_path = os.path.join(dir_path, esdf_file)

            # Use the file name as a variable and call the function
            file_name = os.path.splitext(ply_file)[0]
            total_points, distances = load_esdf_with_distance(file_path)

            esdf_points, esdf_dists = load_esdf_with_distance(esdf_path)

            # Identify points with NaN distances 
            nan_mask = ~np.isnan(distances)
            eval_points = total_points[nan_mask]
            eval_dists = distances[nan_mask]

            # Identify violating points (distance >= 0)
            violating_mask = (eval_dists > 2.001 * voxel_size)
            violating_points = eval_points[violating_mask]

            # Volumen evaluation ESDF points
            vol_mask = (esdf_dists > 0)
            vol_eval_points = esdf_points[vol_mask]
            N_vol_eval_points = vol_eval_points.shape[0]

            N_total_points = total_points.shape[0]
            N_violating_points = violating_points.shape[0]
            violation_rate = (1 * N_violating_points / N_total_points) * 100
            voxel_volume_m3 = voxel_size**3
            max_distance = np.nanmax(distances)
            volume =  N_vol_eval_points * voxel_volume_m3

            folder_results[method] = {"violation_rate": violation_rate,
                                   "max_distance": max_distance,
                                   "volume": volume}

        environment_results[subfolder] = folder_results

    return environment_results 


def process_result_directory(result_folder):
    # Get all environments in the parent folder
    environments = [f.path for f in os.scandir(result_folder) if f.is_dir()]
    environments.sort()
    environments = ["./violation_results/office0"]

    row_folder = [os.path.basename(f) for f in environments]
    row_sigma = ["1e-5", "1e-6"]

    N_rows = len(row_folder) * len(row_sigma)

    col_metric = ["Violation Rate (%)", "Max Violation (mm)", "SFC Volume (m3)"]
    col_method = ["Baseline", "Heuristic", "Certified"]

    N_cols = len(col_metric) * len(col_method)

    row_arr = [row_folder, row_sigma]
    col_arr = [col_metric, col_method]

    row_indices = pd.MultiIndex.from_product(row_arr, names=["Env", "Sigma"])
    col_indices = pd.MultiIndex.from_product(col_arr, names=["Metric", "Method"]) 

    ref_filenames = ["baseline", "heuristic", "certified"]

    results = {}
    cnt = 0
    for env, environment in zip(row_folder, environments):
        if cnt == 1:
            break
        folder_res = process_violation_ply_files(environment, ref_filenames)
        results[env] = folder_res
        logging.info(" ")
        cnt+=1

    pprint.pprint(results)

    row = []
    for env_name, env_data  in results.items():
        for cov_name, cov_data in env_data.items():
            row.append([cov_data['baseline']['violation_rate'], cov_data['heuristic']['violation_rate'], cov_data['certified']['violation_rate'],
                        cov_data['baseline']['max_distance'], cov_data['heuristic']['max_distance'], cov_data['certified']['max_distance'],
                        cov_data['baseline']['volume'], cov_data['heuristic']['volume'], cov_data['certified']['volume']])


    df_res = np.vstack(row)
    logging.info(df_res)
    logging.info(df_res.shape)


    df = pd.DataFrame(df_res, index=row_indices, columns=col_indices)

    # Plot the table
    plt.figure(figsize=(12, len(environments) * 0.5 + 1))
    plt.axis('off')
    table = plt.table(cellText=df.values, colLabels=[f"{c[0]}\n{c[1]}" for c in col_indices], rowLabels=df.index, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.show() 



if __name__ == "__main__":

    # Parse directory container results in argument
    parser = argparse.ArgumentParser(description="Generate Violation Metrics Table")
    parser.add_argument("dir_path",
        help="Path of the results directory containing violation files.")

    args = parser.parse_args()

    logging.basicConfig(format="{asctime} - {levelname} - {message}",
                        style="{",
                        datefmt="%Y-%m-%d %H:%M",
                        level=logging.INFO)

    if not os.path.exists(args.dir_path):
        logging.error("The specified folder does not exist.")
    else:
        logging.info("Directory Path: %s", args.dir_path) 
        process_result_directory(args.dir_path)

