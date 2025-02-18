module VisHelper

using LinearAlgebra, MeshCat 
using GeometryBasics, CoordinateTransformations
using ColorTypes, ColorSchemes, StaticArrays
using Polyhedra
using PlyIO

import MeshCat
using UUIDs: UUID, uuid1

default_colorscheme = reverse(colorschemes[:viridis])

function get_transform(T)
    @assert size(T) == (4,4)
    R = T[1:3,1:3]
    t = T[1:3, 4]
    return Translation(t) ∘ LinearMap(R)
end

function mesh2PointCloud(ply::PlyIO.Ply; cut_ceiling= Inf, color=:auto, colormap=default_colorscheme)
    
    x = ply["vertex"]["x"]
    y = ply["vertex"]["y"]
    z = ply["vertex"]["z"]
    
    inds =  z .<= cut_ceiling
    
    points = map( (x,y,z) -> Point3f(x,y,z), x[inds], y[inds], z[inds])
    
    if color==:auto
        r = ply["vertex"]["red"]
        g = ply["vertex"]["green"]
        b = ply["vertex"]["blue"]

        colors = map( (r,g,b) -> RGB(r/255, g/255, b/255) , r[inds], g[inds], b[inds] )
            
    else
        colors = [color for p in points]
    end
        
    return PointCloud(points, colors )
    
end

function slice_z(pc::PointCloud, zmin=-Inf, zmax=Inf)

    pts = pc.position
    cols = pc.cols
    
    sub_ids = [i for (i, p) in enumerate(pts) if zmin <= p[2] <= zmax]
    sub_pts = pts[sub_ids]
    sub_cols = cols[sub_ids]
    
    return PointCloud(sub_pts, sub_cols)
end

function rotate_ply(ply::PlyIO.Ply, T)
    
    N = length(ply["vertex"])
    homo_pts = transpose(hcat(
        ply["vertex"]["x"],
        ply["vertex"]["y"],
        ply["vertex"]["z"],
        ones(N)
    ))
    
    # do a rotation
    rot_pts = T * homo_pts
    
    # construct new ply file
    new_ply = Ply()
    new_vertex = PlyElement("vertex",
            ArrayProperty("x", rot_pts[1, :]),
            ArrayProperty("y", rot_pts[2, :]), 
            ArrayProperty("z", rot_pts[3, :]),
            ply["vertex"]["intensity"]
    )
    
    push!(new_ply, new_vertex)
    
    return new_ply
    
end

function slice_ply(ply::PlyIO.Ply; z_lims=(-Inf, Inf), intensity_lims=(-Inf, Inf))
    
    il, iu = intensity_lims
    zl, zu = z_lims
    
    intensities = ply["vertex"]["intensity"]
    zs = ply["vertex"]["z"]
    
    N = length(intensities)
    
    good_inds = map( i -> (il <= intensities[i] <= iu) && (zl <= zs[i] <= zu), 1:N)
    
    new_ply = Ply()
    new_vertex = PlyElement("vertex",
            ArrayProperty("x", ply["vertex"]["x"][good_inds]),
            ArrayProperty("y", ply["vertex"]["y"][good_inds]),
            ArrayProperty("z", ply["vertex"]["z"][good_inds]),
            ArrayProperty("intensity", ply["vertex"]["intensity"][good_inds])
    )
    
    push!(new_ply, new_vertex)
    
    return new_ply
end


function sdf2PointCloud(ply::PlyIO.Ply; intensitylims = (-4.0, 4.0), colormap = default_colorscheme)
 
    # filter
    intensities = ply["vertex"]["intensity"]
    il, iu = intensitylims
    inds = map( i-> (il <= i <= iu), intensities) 
    
    # create pts
    pts = map((x,y,z) -> Point3f(x,y,z), 
        ply["vertex"]["x"][inds], 
        ply["vertex"]["y"][inds],
        ply["vertex"]["z"][inds])
    rgbs = map(c -> get(colormap, (c - il) / (iu - il) ), intensities[inds])
        
    pc = MeshCat.PointCloud( pts, rgbs)
    return pc
    
end

function draw_points!(vis, pts; color=RGB(1,0,0), size=0.10f0, kwargs...)
    pc = PointCloud(pts, [color for p in pts])
    draw_pointcloud!(vis, pc; size=size, kwargs...)
end

function draw_pointcloud!(vis, pc; size=0.01f0, cut_ceiling=Inf, alpha=1.0, kwargs...)
    if cut_ceiling == Inf
        return setobject!(vis, pc, PointsMaterial(; size=size, color=RGBA(1,1,1, alpha)))
    else
        inds = [p[3] <= cut_ceiling for p in pc.position]
        modified_pc = PointCloud(pc.position[inds], pc.color[inds])
        return setobject!(vis, modified_pc, PointsMaterial(; size=size, color=RGBA(1,1,1, alpha)) )
    end    
end

function draw_pose!(vis, T; size=1.0f0)
    triad = MeshCat.Triad(size)
    setobject!(vis, triad )
    settransform!(vis, get_transform(T) )
end
    
function draw_polyhedron!(vis, poly; color=RGBA(0,1,0,0.5) )
    mesh = Polyhedra.Mesh(poly )
    draw_mesh!(vis, mesh; color=color)
end
    
function draw_mesh!(vis, mesh; color=RGBA(0,1,0,0.5) )
    setobject!(vis["faces"], mesh, MeshPhongMaterial(;color=color) )
    setobject!(vis["wireframe"], mesh, MeshPhongMaterial(;color=RGB(color), wireframe=true) )
end



struct PlyMesh{TF, TI, TC} <: GeometryBasics.GeometryPrimitive{3, TF}
    coordinates::Union{Nothing, Vector{GeometryBasics.Point{3, TF}}}
    faces::Union{Nothing, Vector{GeometryBasics.TriangleFace{TI}}}
    normals::Union{Nothing, Vector{GeometryBasics.Point{3, TF}}}
    vertexColors::Union{Nothing, Vector{RGB{TC}}}
end

function PlyMesh(ply; color=:auto)
    T = eltype(ply["vertex"]["x"])
    
    # create coords
    x = ply["vertex"]["x"]
    y = ply["vertex"]["y"]
    z = ply["vertex"]["z"]

    coords = map((xi, yi, zi) -> GeometryBasics.Point3f(xi, yi, zi), x, y, z)
    
    # create normals
    nx = ply["vertex"]["nx"]
    ny = ply["vertex"]["ny"]
    nz = ply["vertex"]["nz"]

    normals = map((xi, yi, zi) -> GeometryBasics.Point3f(xi, yi, zi), nx, ny, nz)
    
    # create colors
    if color == :auto
        r = ply["vertex"]["red"]
        g = ply["vertex"]["green"]
        b = ply["vertex"]["blue"]
        colors = map( (ri, gi, bi) -> RGB(ri/255, gi/255, bi/255), r,g,b)
    else
        colors = map( i->color, coords)
    end
    
    # create faces
    faces = GeometryBasics.TriangleFace{Int32}[]
    for (i, quad_face) in enumerate(ply["face"]["vertex_indices"])
        p1, p2, p3, p4 = quad_face
        # +1 is to correct for the offsets
        tri1 = GeometryBasics.TriangleFace{Int32}(p1+1, p2+1, p3+1)
        tri2 = GeometryBasics.TriangleFace{Int32}(p3+1, p4+1, p1+1)
        
        push!(faces, tri1)
        push!(faces, tri2)
    end
    
    PlyMesh(coords, faces, normals, colors)
end


function cut_ceiling(mesh::PlyMesh, h)
    N = length(coordinates(mesh))
        
    coords = coordinates(mesh)
    
    new_inds = zeros(Int, N)
    ind = 1
    for i=1:N
        z = coords[i][3]
        if z <= h
            new_inds[i] = ind
            ind += 1
        end
    end
    
    inds = [i for i=1:N if new_inds[i] > 0]
    
    new_coords = coordinates(mesh)[inds]
    new_normals = normals(mesh)[inds]
    new_colors = mesh.vertexColors[inds]
    
    new_faces = (eltype(mesh.faces))[]
    for face in mesh.faces
        p1, p2, p3 = face
        n1, n2, n3 = new_inds[[p1, p2, p3]]
        
        if (n1 > 0 && n2 > 0 && n3 > 0)
            push!(new_faces, TriangleFace(n1, n2, n3) )
        end
    end
            
    return PlyMesh(new_coords, new_faces, new_normals, new_colors)
end
            
            

GeometryBasics.coordinates(mesh::PlyMesh) = (mesh.coordinates)
GeometryBasics.faces(mesh::PlyMesh) = (mesh.faces)
GeometryBasics.texturecoordinates(mesh::PlyMesh) = nothing
GeometryBasics.normals(mesh::PlyMesh) = (mesh.normals)

# hijack the lower methods to draw the mesh
function MeshCat.lower(mesh::PlyMesh)
    
    attributes = Dict{String, Any}(
        "position" => MeshCat.lower(convert(Vector{Point3f}, decompose(Point3f, mesh))),
    )
    
    # get the colors
    attributes["color"] = MeshCat.lower(convert(Vector{RGB{Float32}}, mesh.vertexColors))
        
    Dict{String, Any}(
        "uuid" => string(uuid1()),
        "type" => "BufferGeometry",
        "data" => Dict{String, Any}(
            "attributes" => attributes,
            "index" => MeshCat.lower(decompose(GLTriangleFace, mesh))
        )
    )
end


# provide a method to draw the mesh
function draw_mesh!(vis, plymesh::PlyMesh)
    mat = MeshCat.MeshPhongMaterial(; vertexColors = 1)
    setobject!(vis, plymesh, mat)
end



end

