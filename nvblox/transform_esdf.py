import argparse
import numpy as np


def load_esdf_with_intensity(file_path):
    """
    Load an ESDF .ply file and extract points (x, y, z) and intensity.
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

    points = data[:, :3]
    intensity = data[:, 3]
    return points, intensity, lines[:data_start_idx]  # Return header


def apply_transformation(points, transformation_matrix):
    """
    Apply a 4x4 homogeneous transformation matrix to (x, y, z) points.
    """
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (transformation_matrix @ homogeneous_points.T).T
    return transformed_points[:, :3]


def save_transformed_esdf(file_path, transformed_points, intensity, header):
    """
    Save the transformed points and intensity to a .ply file in ASCII format.
    """
    # Combine transformed points and intensity
    data = np.hstack((transformed_points, intensity[:, np.newaxis]))

    with open(file_path, "w") as f:
        # Write the original header
        for line in header:
            f.write(line)
        # Write the transformed points and intensity
        np.savetxt(f, data, fmt="%.6f")
    print(f"Transformed ESDF saved to {file_path} with {transformed_points.shape[0]} points.")


def main():
    parser = argparse.ArgumentParser(description="Apply a 4x4 transformation to an ESDF .ply file.")
    parser.add_argument("input_file", help="Path to the input ESDF .ply file (x, y, z, intensity).")
    parser.add_argument("output_file", help="Path to save the transformed ESDF .ply file.")
    parser.add_argument("transform_file", help="Path to the text file containing the 4x4 transformation matrix.")
    args = parser.parse_args()

    try:
        with open(args.transform_file, "r") as f:
            transformation_line = f.readline().strip()
        transformation_values = list(map(float, transformation_line.split()))
        if len(transformation_values) != 16:
            raise ValueError("Transformation matrix must have 16 values.")
        transformation_matrix = np.array(transformation_values).reshape(4, 4)
    except Exception as e:
        raise ValueError(f"Error reading transformation matrix: {e}")

    # Load the input ESDF file
    points, intensity, header = load_esdf_with_intensity(args.input_file)

    # Apply the transformation matrix
    transformed_points = apply_transformation(points, transformation_matrix)

    # Update header with the correct number of points
    for i, line in enumerate(header):
        if line.startswith("element vertex"):
            header[i] = f"element vertex {transformed_points.shape[0]}\n"

    # Save the transformed points and intensity
    save_transformed_esdf(args.output_file, transformed_points, intensity, header)


if __name__ == "__main__":
    main()
