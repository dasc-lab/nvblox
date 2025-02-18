module SE3

using LinearAlgebra
using StaticArrays

rotation(T::SMatrix{4,4}) = T[SOneTo(3), SOneTo(3)]
translation(T::SMatrix{4,4}) = T[SOneTo(3), 4]

function skew(v::SVector{3, F}) where {F}
    return @SMatrix [
        [0 ;; -v[3];; v[2]];
        [v[3] ;; 0 ;; -v[1]];
        [-v[2] ;; v[1] ;; 0]]
end

function hat(v::SVector{6, F}) where {F}
    ρ = SVector{3, F}(v[1], v[2], v[3])
    θ = SVector{3, F}(v[4], v[5], v[6])
    
    return SMatrix{4,4, F, 16}( 
        [
            [skew(θ) ;; ρ];
            zeros(1, 4)
        ] )
end
hat(v) = hat(SVector{6}(v) )

function Exp(v::SVector{6})
    τ = hat(v)
    return exp(τ )
end
Exp(v) = Exp(SVector{6}(v) )


function action(T::SMatrix{4,4}, p::SVector{3})
    return rotation(T) * p + translation(T)
end
action(T, p) = action(SMatrix{4,4}(T), SVector{3}(p) )

function action_jacobian(T::SMatrix{4,4}, p::SVector{3})
    R = rotation(T)
    J = [R ;; -R * skew(p)]
    return SMatrix{3, 6}(J )
end
action_jacobian(T, p) = action_jacobian(SMatrix{4,4}(T), SVector{3}(p) )
    
end
