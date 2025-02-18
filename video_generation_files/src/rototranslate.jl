module Rototranslate

using LinearAlgebra, StaticArrays, Polyhedra, GeometryBasics
using MeshCat

include("./se3.jl")
using .SE3


function rototranslate(hs::HalfSpace, pose::SMatrix{4,4})
    R = pose[1:3, 1:3]
    t = pose[1:3, 4]
    
    a_new = R * hs.a
    b_new = hs.β + a_new' * t
    return HalfSpace(a_new/norm(a_new), b_new/norm(a_new) )
end
    
function rototranslate(poly::HRep, pose::SMatrix{4,4})
    new_hs = [rototranslate(hs, pose) for hs in halfspaces(poly)]
    return polyhedron(hrep(new_hs) )
end
    
# function facet_representation(poly)
#     corner_sets = Vector{SVector{3, Float64}}[]
#     halfspaces_ = HalfSpace{Float64, Vector{Float64}}[]
#     for hs in halfspaces(poly)
#         hp = Polyhedra.hyperplane(hs)
#         intersect_ = intersect(poly, hp)
#         facet = vrep(intersect_)
#         corners = collect(SVector{3}.(points(facet)))
#         if length(corners) > 0
#             push!(corner_sets, corners)
#             push!(halfspaces_, hs)
#         end
#     end
#     return corner_sets, halfspaces_
# end

function facet_representation(poly::Polyhedron)
    mesh = Polyhedra.Mesh{3}(poly)
    Polyhedra.fulldecompose!(mesh)
    
    corner_sets_ = Vector{SVector{3, Float64}}[]
    halfspaces_ = HalfSpace{Float64, Vector{Float64}}[]
    
    for face in faces(mesh)
        cs = coordinates(mesh)[face]
        n = normalize(normals(mesh)[face[1]])
        
        # n' * x >= n' p
        hs = HalfSpace(n, n' * cs[1])
       
        # push
        push!(corner_sets_, cs)
        push!(halfspaces_, hs)
    end
    
    return corner_sets_, halfspaces_
end


function rototranslate(poly::HRep, pose::SMatrix{4,4}, ΣT; n_std=3)
    
    # first get the facet representation
    corner_sets, half_spaces = facet_representation(poly)
    
    new_halfspaces = (eltype(half_spaces))[]
    
    for i=1:length(half_spaces)
        corners = corner_sets[i]
        hs = half_spaces[i]

        # get the rotated normal vector
        rot_hs = rototranslate(hs, pose)
        n = rot_hs.a
        
        max_r = 0.0
        for corner in corners
            new_corner = SE3.action(pose, corner)
            J = SE3.action_jacobian(pose, corner)
            Σp = Symmetric(J * ΣT * J')
            r2 = n_std * sqrt(eigmax(Σp))
            r = get_height_along_normal(Σp, n; n_std=n_std)
            max_r = max(max_r, r)
        end
        
        # now compute the new halfspace
        new_hs = HalfSpace(n, rot_hs.β - max_r)
        push!(new_halfspaces, new_hs)
    end
        
    new_poly = polyhedron(hrep(new_halfspaces) )
    return new_poly
end

function rototranslate(pointcloud::PointCloud, pose::SMatrix{4,4})
    pts = [SE3.action(pose, p) for p in pointcloud.position]
    return PointCloud(pts, pointcloud.color)
end

function tangent_point(Σ, n; n_std=3)
    # x' Sinv Sinv x == n_std^2

    # v = Sinv x

    # given a unit vector v
    # x = S * v is a point on the ellipsoid

    # the normal vector of the tangent x is proportional to
    # inv(Σ) * x
    # inv(Σ) * S * v
    # Sinv * Sinv * S * v
    # Sinv * v

    # so we want Sinv * v to be parallel to n

    S = sqrt(Σ)
    v = normalize(S * n)
    x0 = n_std * S * v
end

function get_height_along_normal(Σ, n; n_std=3)
    x0 = tangent_point(Σ, n; n_std=n_std)
    h = dot(normalize(n), x0)
    return h
end


end

