using DifferentialEquations, Random, Distributions, LinearAlgebra, GLMakie, Observables, FastGaussQuadrature

mutable struct Particle
    idp::Int64              #ID of the particle
    massp::Float64          #Mass of the particle
    radiusp::Float64        #Radius of particles

    th_v0p::Float64         # Velocidad angular inicial
    th_0p::Float64          # Desplazamiento angular inicial  

    posp::Vector{Float64}   #Position vector of the particle
    velp::Vector{Float64}   #Velocity vector of the particle
end

mutable struct Manifold
    n_params::Int
    parametrization::Function
end

function metric_tensor(manifold::Manifold, u::Vector{Float64}, ε::Float64=1e-5)
    n_params = manifold.n_params
    param_func = manifold.parametrization
    ∂param_∂u = zeros(Float64, n_params, length(param_func(u...)))
    
    for i in 1:n_params
        u_forward = copy(u)
        u_forward[i] += ε
        u_backward = copy(u)
        u_backward[i] -= ε
        ∂param_∂u[i, :] = (param_func(u_forward...) - param_func(u_backward...)) / (2ε)
    end
    
    g = ∂param_∂u * transpose(∂param_∂u)
    return g
end

function christoffel_symbols(manifold::Manifold, u::Vector{Float64}, ε::Float64=1e-5)
    n_params = manifold.n_params
    g = metric_tensor(manifold, u, ε)
    Γ = zeros(Float64, n_params, n_params, n_params)
    
    for k in 1:n_params
        for i in 1:n_params
            for j in 1:n_params
                Γ[k, i, j] = 0.5 * sum(pinv(g)[k, l] * (∂g_ij_l(manifold, u, i, j, l, ε)) for l in 1:n_params)
            end
        end
    end
    
    return Γ
end

function ∂g_ij_l(manifold::Manifold, u::Vector{Float64}, i::Int, j::Int, l::Int, ε::Float64=1e-5)
    u_forward = copy(u)
    u_backward = copy(u)
    u_forward[l] += ε
    u_backward[l] -= ε
    g_forward = metric_tensor(manifold, u_forward, ε)
    g_backward = metric_tensor(manifold, u_backward, ε)
    
    return (g_forward[i, j] - g_backward[i, j]) / (2 * ε)
end

function covariant_derivative(manifold::Manifold, particle::Particle, t::Float64, dt::Float64)
    u = particle.posp
    du = particle.velp

    # Proyectar la velocidad actual al espacio tangente antes de usarla
    du = project_to_tangent(du, manifold, u)

    Γ = christoffel_symbols(manifold, u)
    
    acc = zeros(Float64, length(u))
    for k in 1:length(u)
        for i in 1:length(u)
            for j in 1:length(u)
                acc[k] -= Γ[k, i, j] * du[i] * du[j]
            end
        end
    end
    
    return acc
end

function arc_length_gauss_legendre(gamma, t1, t2, n::Integer)
    # Obtener nodos y pesos de Gauss-Legendre
    x, w = gausslegendre(n)
    
    # Transformar los nodos del intervalo [-1, 1] a [t1, t2]
    t_vals = 0.5 * ((t2 - t1) .* x .+ (t2 + t1))  # Escala y traslada los nodos a [t1, t2]
    
    # Derivada de la curva en los nodos t_vals
    integrand(t) = norm(differentiate(gamma, t))  # Diferenciar gamma para obtener la velocidad
    
    # Evaluar el integrando en los nodos y calcular la longitud de arco
    arc_length = 0.5 * (t2 - t1) * dot(w, integrand.(t_vals))
    
    return arc_length
end

function check_collision(p1::Particle, p2::Particle, manifold::Manifold)
    gamma = manifold.parametrization
    t1 = p1.param  
    t2 = p2.param
    dist_arc = arc_length_gauss_legendre(gamma, t1, t2, 20) 
    return dist_arc < (p1.radiusp + p2.radiusp)  
end

# Proyección al espacio tangente de la variedad
function project_to_tangent(velocity, manifold::Manifold, position::Vector{Float64}, ε::Float64=1e-5)
    # Obtención de la métrica en el punto actual
    g = metric_tensor(manifold, position, ε)
    
    # Proyección ortogonal al espacio tangente
    tangent_velocity = velocity - (velocity' * pinv(g) * velocity) * velocity / norm(velocity)
    return tangent_velocity
end

# Transporte paralelo de un vector a lo largo de una curva
function parallel_transport(manifold::Manifold, v::Vector{Float64}, u_start::Vector{Float64}, u_end::Vector{Float64}, ε::Float64=1e-5)
    # Calcula los símbolos de Christoffel en u_start
    Γ = christoffel_symbols(manifold, u_start)
    
    # Paso de integración
    dt = 1e-3 
    u = copy(u_start)
    transported_v = copy(v)

    # Vector diferencial de la curva entre u_start y u_end
    du = (u_end - u_start) * dt / norm(u_end - u_start)

    while norm(u - u_end) > ε
        # Resuelve la ecuación del transporte paralelo
        dv = zeros(Float64, length(v))
        for k in 1:length(u)
            for i in 1:length(u)
                for j in 1:length(u)
                    dv[k] -= Γ[k, i, j] * du[i] * transported_v[j]
                end
            end
        end
        
        # Actualiza el vector transportado
        transported_v += dv * dt
        u += du  # Avanza a lo largo de la curva
    end
    
    return transported_v
end

# Función para manejar colisiones con transporte paralelo
function handle_collision!(p1::Particle, p2::Particle, manifold::Manifold)
    normal = normalize(p2.position - p1.position)

    v1_normal = dot(p1.velocity, normal) * normal
    v2_normal = dot(p2.velocity, normal) * normal
    v1_tangent = p1.velocity - v1_normal
    v2_tangent = p2.velocity - v2_normal

    v1_tangent_before = v1_tangent
    v2_tangent_before = v2_tangent

    v1_normal_new = (p1.mass - p2.mass) / (p1.mass + p2.mass) * v1_normal + 2 * p2.mass / (p1.mass + p2.mass) * v2_normal
    v2_normal_new = (p2.mass - p1.mass) / (p1.mass + p2.mass) * v2_normal + 2 * p1.mass / (p1.mass + p2.mass) * v1_normal

    p1.velocity = parallel_transport(manifold, v1_normal_new + v1_tangent, p1.position, p2.position)
    p2.velocity = parallel_transport(manifold, v2_normal_new + v2_tangent, p2.position, p1.position)

    v1_tangent_after = p1.velocity - dot(p1.velocity, normal) * normal
    v2_tangent_after = p2.velocity - dot(p2.velocity, normal) * normal

    if norm(v1_tangent_before - v1_tangent_after) > 1e-5 || norm(v2_tangent_before - v2_tangent_after) > 1e-5
        println("Advertencia: la componente tangencial ha cambiado después de la colisión.")
    end
end

function project_to_manifold(pos::Vector{Float64}, manifold::Manifold)
    # Define la parametrización del manifold como una función en términos de t
    param = manifold.parametrization
    
    # Define el intervalo de t a buscar
    t_min, t_max = 0.0, 2π  # Ajustar según el manifold

    # Minimizar la distancia entre pos y la curva parametrizada por manifold
    opt_result = optimize(t -> norm(param(t) - pos), t_min, t_max)
    
    # Obtener el valor óptimo de t
    t_proj = opt_result.minimizer
    
    # Devuelve el punto proyectado sobre la curva
    return param(t_proj)
end

function generate_random_positions(manifold::Manifold, p::Float64, n::Int64)
    positions = []   # Inicializa el vector de posiciones
    angles = []      # Inicializa el vector de ángulos

    param_func = manifold.parametrization
    
    # Generar n posiciones
    while length(positions) < n
        ϕ_0 = rand(Uniform(0, 2π))  # Genera un ángulo aleatorio entre 0 y 2π
        
        R_value = param_func(ϕ_0)  # Esto evalúa la función con el ángulo ϕ_0

        # Calcula la posición utilizando el valor numérico obtenido de R(ϕ_0)
        pos = [R_value * cos(ϕ_0), R_value * sin(ϕ_0)]  # Usa coordenadas polares (x, y)
        
        overlapping = false
        for existing_pos in positions
            if norm(pos - existing_pos) < 2 * p
                overlapping = true
                break
            end
        end
        
        # Si no se solapan, añade la posición al vector de posiciones y el ángulo al vector de ángulos
        if !overlapping
            push!(positions, pos)
            push!(angles, ϕ_0)
        end
    end

    return positions, angles
end

function evolve_system_suzuki_trotter!(manifold::Manifold, particles::Vector{Particle}, t::Float64, dt::Float64, n_steps::Int)
    # Coeficientes
    w1 = 1 / (2 - 2^(1/3))
    w2 = 1 - 2 * w1
    v1 = w1
    v2 = w2
    v3 = w1

    for step in 1:n_steps
        # Paso 1: Evolución de la velocidad (medio paso) - v1
        for p in particles
            acc = covariant_derivative(manifold, p, t, dt)
            p.velocity += acc * (v1 * dt / 2.0)
        end

        # Paso 2: Evolución de la posición (paso completo) - w1
        for p in particles
            p.position += p.velocity * (w1 * dt)
        end

        # Paso 3: Evolución de la velocidad (medio paso) - v2
        for p in particles
            acc = covariant_derivative(manifold, p, t, dt)
            p.velocity += acc * (v2 * dt / 2.0)
        end

        # Paso 4: Evolución de la posición (paso completo) - w2
        for p in particles
            p.position += p.velocity * (w2 * dt)
        end

        # Paso 5: Evolución de la velocidad (medio paso) - v3
        for p in particles
            acc = covariant_derivative(manifold, p, t, dt)
            p.velocity += acc * (v3 * dt / 2.0)
        end

        # Paso 6: Evolución de la posición (paso completo) - w1
        for p in particles
            p.position += p.velocity * (w1 * dt)
        end

        # Paso 7: Evolución de la velocidad (segundo medio paso) - v1
        for p in particles
            acc = covariant_derivative(manifold, p, t, dt)
            p.velocity += acc * (v1 * dt / 2.0)
        end

        # Verificar y manejar colisiones entre partículas
        for i in 1:length(particles)-1
            for j in i+1:length(particles)
                if check_collision(particles[i], particles[j])
                    handle_collision!(particles[i], particles[j], manifold)
                end
            end
        end
        t += dt
    end

end
#------------------------------------------------------------------------#

a = 1.0  
b = 0.5  
R(θ) = a + b * cos(θ)
limacon_manifold = Manifold(1, R)

nstep = 10000
titer = 50.0
tstep = 1e-4
p = 0.05
n = 12

prtcl = Vector{Particle}(undef, n)
rndm_pstns, ngls = generate_random_positions(limacon_manifold, p, n)

for i in 1:n
    ϕ_v0 = rand(Uniform(-1e11, 1e11))
    prtcl[i] = Particle(i, 1.00, p, ϕ_v0, ngls[i], rndm_pstns[i], [-ϕ_v0 * sin(ngls[i]); ϕ_v0 * cos(ngls[i])])
end

positions = Observable(Point2f.(getproperty.(prtcl, :posp)))
f = Figure(resolution=(750,750))
ax = Axis(f[1,1], aspect=1)
scatter!(ax, positions, color=:gray, markersize=2 * p, markerspace=:clip)
limits!(ax, -1, 2, -1.5 , 1.5)
f

for i in 1:titer
    evolve_system_suzuki_trotter!(limacon_manifold, prtcl, titer, tstep, nstep)
    positions[] = Point2f.(getproperty.(prtcl, :posp))
end
