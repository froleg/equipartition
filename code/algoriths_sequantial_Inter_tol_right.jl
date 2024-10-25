using BenchmarkTools
using XLSX, DataFrames
using LinearAlgebra
using NonlinearSolve


struct PlPoint
    x::Float64
    y::Float64
end

function poly_length(points)
    return [norm(points[i+1]-points[i]) for i in 1: length(points)-1]
end

function poly_length_pt(points)
    return [norm([points[i+1].x, points[i+1].y]-[points[i].x, points[i].y]) for i in 1: length(points)-1]
end

function initial_radius(strt, stp, n_part, control)
    uniform_step =  (stp - strt)/n_part
    uniform_params = [i*uniform_step+strt for i in 0:(n_part)]
    uniform_point_array = [[compute_bernstein(control[1,:], par), compute_bernstein(control[2,:], par)] for par in uniform_params]
    initial_len_array = poly_length(uniform_point_array)
    return mean(initial_len_array)
end

function compute_bernstein(P, t)
    n = Integer(length(P)-1)
    return sum([binomial(n,i)*t^i*(1-t)^(n-i) for i in 0:n].*P)
end

function find_root(u0, un, xo, yo, r)
    fi(u, p) = ((68679894317400063*u^6-265149428061437934*u^5+374924668978593750*u^4-352406670841741280*u^3+234750130576687095*u^2-51003265779970866*u+5629499534213120)/1125899906842624-xo)^2 + (u*(2328361007350546385*u^5-6885103110324014142*u^4+7039126217580085080*u^3-2661627379775963040*u^2+40532396646334440*u+135107988821114880)/2251799813685248-yo)^2 - r^2
    f(u, p) = u^4 - 2.0
    uspan = (u0, un)
    prob = IntervalNonlinearProblem(fi, uspan)
    sol = solve(prob)
    return sol.u
end

function eval_poly_pt(points, parms, n_pt, direction, radius, cntr )
    if direction
      start = 1
      stp = n_pt
      step = 1
      u = parms[length(parms)]
    else
      start = length(parms)
      stp = start - n_pt +1
      step = -1
      u = parms[1]
    end
    for i in start:step:stp
      xc, yc = points[i].x, points[i].y
      #println(xc, " ", yc)
      t = find_root(parms[i], u, xc, yc, radius)
      #println("t= ",t)
      parms[i+step] = t
      points[i+step] = PlPoint(compute_bernstein(cntr[1,:], t), compute_bernstein(cntr[2,:], t))
    end
end

# sequantial version of algorithm
function calc_seq(start_par, end_par, cntr_poly; tol = 0.000001, n_part = 1047)
    n_iter = 0
    points_arr = Vector{PlPoint}(undef,n_part+1)
    points_arr[1] = PlPoint(compute_bernstein(cntr_poly[1,:], start_par), compute_bernstein(cntr_poly[2,:], start_par))
    points_arr[n_part+1] = PlPoint(compute_bernstein(cntr_poly[1,:], end_par), compute_bernstein(cntr_poly[2,:], end_par))
    par_arr = Vector{Float64}(undef,n_part+1)
    par_arr[1] = start_par
    par_arr[n_part+1] = end_par
    radius = initial_radius(start_par, end_par, n_part, cntr_poly)
    while true
        eval_poly_pt(points_arr, par_arr, n_part-1, true, radius, cntr_poly)
        n_iter+=1
        control_len = poly_length_pt(points_arr[n_part:end])
        sg = sign(par_arr[end] - par_arr[end-1])
  #      println(control_len)
        if abs(control_len[1] - radius) > tol
            radius = (sg*control_len[1]+(n_part-1)*radius)/n_part
  #          println("r = ", radius)
        else
            break
        end
    end
    return n_iter, radius
end

const start_par = 0.
const end_par= 1.
const cntr_pt = [5. -2.55 3.8  8.4  17.8  21.5 13.7;  0. 10. 21.2 -25.5 19.2 3.4 -1.6]


#t = @benchmark calc_seq(start_par, end_par, cntr_pt, tol = 0.000001, n_part = 1000000)
#dump(t)
#m1 = median(t)
#println("median m1 ", m1)
#println("time ", t.times)

preprocess_trial(id::AbstractString, num_it::Integer, rads::AbstractFloat, tol::AbstractFloat, n::Integer) =
           (id=id,
           tolerance = tol,
           n = n,
           num_it = num_it,
           radius = rads)

output = DataFrame()

parts = [(27, 0.001), (27, 0.0005), (27, 0.0001), (27, 0.00005), (27, 0.00001), (27, 0.000005), (27, 0.000001), (27, 0.0000005), (27, 0.0000001), (30, 0.001), (30, 0.0005), (30, 0.0001), (30, 0.00005), (30, 0.00001), (30, 0.000005), (30, 0.000001), (30, 0.0000005), (30, 0.0000001), (40, 0.001), (40, 0.0005),(40, 0.0001), (40, 0.00005), (40, 0.00001), (40, 0.000005), (40, 0.000001), (40, 0.0000005), (40, 0.0000001), (50, 0.001), (50, 0.0005), (50, 0.0001), (50, 0.00005), (50, 0.00001), (50, 0.000005), (50, 0.000001), (50, 0.0000005), (50, 0.0000001), (75, 0.001), (75, 0.0005), (75, 0.0001), (75, 0.00005), (75, 0.00001), (75, 0.000005), (75, 0.000001), (75, 0.0000005), (75, 0.0000001),  (100, 0.001), (100, 0.0005), (100, 0.0001), (100, 0.00005), (100, 0.00001), (100, 0.000005), (100, 0.000001), (100, 0.0000005), (100, 0.0000001), (100, 0.00000005), (100, 0.00000001),(100, 0.000000005),(100, 0.000000001), (150, 0.0001), (150, 0.00005), (150, 0.00001), (150, 0.000005), (150, 0.000001), (150, 0.0000005), (150, 0.0000001), (150, 0.00000005),(150, 0.00000001), (150, 0.000000005),(150, 0.000000001), (150, 0.0000000005),(150, 0.0000000001), (250, 0.0001), (250, 0.00005), (250, 0.00001), (250, 0.000005), (250, 0.000001), (250, 0.0000005), (250, 0.0000001),(250, 0.00000005),(250, 0.00000001),(250, 0.000000005),(250, 0.000000001),(250, 0.0000000005),(250, 0.0000000001),(250, 0.00000000005),(250, 0.00000000004),(250, 0.00000000002),(250, 0.00000000001), (500, 0.00005), (500, 0.00001), (500, 0.000005), (500, 0.000001), (500, 0.0000005),(500, 0.0000001),(500, 0.00000005),(500, 0.00000001),(500, 0.000000005),(500, 0.000000001),(500, 0.0000000005), (500, 0.0000000001), (500, 0.00000000005), (500, 0.00000000004),(500, 0.00000000002),(500, 0.00000000001), (500, 0.000000000001), (755, 0.00005), (755, 0.00001), (755, 0.000005), (755, 0.000001), (755, 0.0000005),(755, 0.0000001),(755, 0.00000005),(755, 0.00000001),(755, 0.000000005),(755, 0.000000001),(755, 0.0000000005),(755, 0.0000000001),(755, 0.00000000005),(755, 0.00000000004),(755, 0.00000000002),(755, 0.00000000001), (755, 0.000000000001), (1000, 0.00005), (1000, 0.00001), (1000, 0.000005), (1000, 0.000001), (1000, 0.0000005),(1000, 0.0000001),(1000, 0.00000005),(1000, 0.00000001),(1000, 0.000000005),(1000, 0.000000001),(1000, 0.0000000005),(1000, 0.0000000001),(1000, 0.00000000005),(1000, 0.00000000004),(1000, 0.00000000002),(1000, 0.00000000001)]
tols = [ 0.001, 0.0001,  0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001, 0.0000000001]
println(typeof(parts))
for  tl in tols
    for n_p in 27:1000
        toler = tl
        println(n_p, " ", toler)
    #    @btime calc_seq(start_par, end_par, cntr_pt, tol = toler, n_part = n_p)
        ni, rad = calc_seq(start_par, end_par, cntr_pt, tol = toler, n_part = n_p)
        println(" ni=", ni, " radii=", rad)
        p_t =  preprocess_trial("n = $n_p, tol = $toler", ni, rad, toler, n_p)
        push!(output, p_t)
    end
end

XLSX.writetable("report_seq_tol_right_contin27-1000.xlsx", output)
