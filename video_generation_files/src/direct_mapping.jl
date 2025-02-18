module DirectMapping

using LinearAlgebra, MeshCat, GeometryBasics, CoordinateTransformations, ColorTypes, ColorSchemes
using DelimitedFiles, Images, JSON, FileIO
using StaticArrays, Polyhedra
using Printf
using DecompUtil, LinearAlgebra, StaticArrays, Polyhedra


const default_colorscheme = reverse(colorschemes[:viridis])

struct Camera{F}
    w::Int
    h::Int
    fx::F
    fy::F
    cx::F
    cy::F
    scale::F
end

function camera_K_matrix(camera) 
    return  @SMatrix [[ camera.fx ;; 0;; camera.cx]; [0 ;; camera.fy ;; camera.cy]; [ 0 ;; 0 ;; 1]]
end

function read_camera_json(camera_params_path)
    f = JSON.parsefile(camera_params_path)
    return Camera{Float64}(
        f["camera"]["w"],
        f["camera"]["h"],
        f["camera"]["fx"],
        f["camera"]["fy"],
        f["camera"]["cx"],
        f["camera"]["cy"],
        f["camera"]["scale"] 
        )
end

function get_camera_fov_polyhedron(camera::Camera, max_depth = 5.0)
    w = camera.w
    h = camera.h
    
    invK = inv(camera_K_matrix(camera))
    corners = [
        [0;; 0;; w;;  w;; 0;;];
        [0;; 0;; 0;;  h;; h;;];
        [0;; 1;; 1;;  1;; 1;;]
    ]
    
    fov_corners = max_depth * (invK * corners)
    
    return polyhedron(Polyhedra.vrep(fov_corners') )
end


function read_trajectory(traj_file)
    M = readdlm(traj_file, ' ', Float64, '\n')
    return map( v -> SMatrix{4,4}(collect(reshape(v, 4,4)')), eachrow(M) )
end

function depth_image_path(index, base_path)
    return base_path * (@sprintf("/results/depth%06d.png", index-1) )
end
function color_image_path(index, base_path)
    return base_path * (@sprintf("/results/frame%06d.jpg", index-1) )
end
function trajectory_path(base_path)
    return base_path * "/traj.txt"
end  
function camera_file_path(base_path)
    return base_path * "../cam_params.json"
end  

function get_depth_pointcloud_fast(index, base_path, camera) 

    depth_image = load(depth_image_path(index, base_path) )::Matrix{Gray{FixedPointNumbers.N0f16}}
    
    # get the camera matrix
    invK = inv(camera_K_matrix(camera))
    
    # extract rows and cols
    rows = camera.h
    cols = camera.w

    # create all the depth points
    # points = Point3f[]
    points = Vector{Point3f}(undef, rows * cols)
    
    for (i, ind) in enumerate(CartesianIndices((1:rows, 1:cols)))
        u = ind[2] # col = x
        v = ind[1] # row = y
        d = depth_image[ind]
    
        z = (2^16 - 1) * d / camera.scale
    
            pt =  (invK * (@SVector [u*z, v*z, z]) )
            points[i] = pt

    end
    
    return PointCloud(points)# , colors)
end



# colorby can be :z, :image
function get_depth_pointcloud(index, base_path;
        subsample=1, colorby=:z, colormap=default_colorscheme)
    
    depth_image = load(depth_image_path(index, base_path) )
    camera = read_camera_json(camera_file_path(base_path) )
    
    # get the camera matrix
    K = camera_K_matrix(camera)
    
    # convert to actual depths
    depths = map( d -> (2^16 - 1) * d.val / camera.scale, depth_image)
    
    # extract rows and cols
    rows = camera.h
    cols = camera.w

    # create all the depth points
    points = Point3f[]
    colors = RGB[]
        
    N = length(1:subsample:rows) * length(1:subsample:cols)
    sizehint!(points, N)
    sizehint!(colors, N)
    
    skip_inds = CartesianIndex[]
    
    for ind in CartesianIndices((1:subsample:rows, 1:subsample:cols))
        u = ind[2] # col = x
        v = ind[1] # row = y
    
        z = depths[ind]
        
        if z > 0 
            pt = z * (K \ [u, v, 1])
            push!(points, Point3f(pt) )
        else
            push!(skip_inds, ind)
        end

    end
    
    if colorby == :z
       max_depth = maximum(depths)
       for ind in CartesianIndices((1:subsample:rows, 1:subsample:cols))
            if !(ind ∈ skip_inds) 
                z = depth_image[ind] * camera.scale
                push!(colors, get(colormap, z/max_depth) )
            end
        end
    elseif colorby == :image
        color_image = load(color_image_path(index, base_path) )
        for ind in CartesianIndices((1:subsample:rows, 1:subsample:cols) )
            if !(ind ∈ skip_inds)
                push!(colors, color_image[ind] )
            end
        end
    else
        @assert colorby ∈ [:z, :image]
    end
    
    return PointCloud(points, colors)
end


# get SFC
function get_sfc_polytope(seed, pointcloud::PointCloud, camera::Camera;
        max_depth=5.0,
        dilation_radius = 0.01,
        max_poly = 200)
    
    bbox = [max_depth, max_depth, max_depth]
    
    # get the decomp sfc
    surfs = seedDecomp(seed, pointcloud.position, bbox, dilation_radius, max_poly)
    
    # normalize
    A, b = constraints_matrix(surfs)
    for i=1:size(A, 1)
        a = A[i, :]
        n = norm(a)
        A[i, :] = a / n
        b[i] = b[i] / n
    end
    
    # construct sfc
    decomp_sfc = polyhedron(hrep(A, b))
    
    # get the camera fov
    fov = get_camera_fov_polyhedron(camera )
    
    # do an intersection
    poly = intersect(decomp_sfc, fov)
    
    # return a nice polyhedron
    return polyhedron(MixedMatHRep(hrep(poly))) 
end


end
