

using BenchmarkTools


using LinearAlgebra
using NonlinearSolve
using Base.Threads


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

function eval_poly!(points, parms, bound_par, n_pt, direction, radius, cntr)
  if direction
    start = 1
    stp = n_pt-1
    step = 1
  else
    start = n_pt
    stp = 2
    step = -1
  end
  u = bound_par
  for i in start:step:stp
    @inbounds xc, yc = points[i].x, points[i].y
    #println(xc, " ", yc)
    @inbounds t = find_root(parms[i], u, xc, yc, radius)
    #println("t= ",t)
    @inbounds parms[i+step] = t
    @inbounds points[i+step] = PlPoint(compute_bernstein(cntr[1,:], t), compute_bernstein(cntr[2,:], t))
  end
end
# two threds version of algorithm
function calc_th(start_par, end_par, cntr_poly; tol = 0.000001, n_part = 1047)
  n_iter = 0
  first_half = div((n_part),2) + 1
  second_half = n_part - first_half + 1
  bounds = [first_half, second_half]
  points_f = Vector{PlPoint}(undef, first_half )
  points_f[1] = PlPoint(compute_bernstein(cntr_poly[1,:], start_par), compute_bernstein(cntr_poly[2,:], start_par))
  points_s = Vector{PlPoint}(undef, second_half)
  points_s[second_half] = PlPoint(compute_bernstein(cntr_poly[1,:], end_par), compute_bernstein(cntr_poly[2,:], end_par))
  params_f = Vector{Float64}(undef, first_half)
  params_f[1] = start_par
  params_s = Vector{Float64}(undef, second_half)
  params_s[second_half] = end_par
  radius = initial_radius(start_par, end_par, n_part, cntr_poly)
  end_params = [end_par, start_par]
  bounds = [first_half, second_half]
  directs = [true, false]
  pt_tup = (points_f, points_s)
  pr_tup = (params_f, params_s)
  while true
    @sync for i in 1:2
       @spawn eval_poly!(pt_tup[i], pr_tup[i], end_params[i], bounds[i], directs[i], radius, cntr_poly)
    end
    n_iter+=1
    control_len = poly_length_pt([points_f[end], points_s[1]])
    sg = sign(params_s[1] - params_f[end])
#      println(control_len)
    if abs(control_len[1] - radius) > tol
        radius = (sg*control_len[1]+(n_part-1)*radius)/n_part
#          println("r = ", radius)
    else
        break
    end
  end
#    println("iter = ", n_iter)
  return cat(points_f, points_s, dims = (1,1)), cat(params_f, params_s, dims = (1,1))
end

const start_par = 0.
const end_par= 1.
const cntr_pt = [5. -2.55 3.8  8.4  17.8  21.5 13.7;  0. 10. 21.2 -25.5 19.2 3.4 -1.6]

println("threads ", nthreads())
@btime calc_th(start_par, end_par, cntr_pt, tol = 0.000001, n_part = 1000000)
