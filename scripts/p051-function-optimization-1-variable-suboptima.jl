using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p051-function-optimization-1-variable-suboptima"
using CairoMakie
using LaTeXStrings
using ForwardDiff

# optimization method for 1 variable continuous (differentiable) function
# 1. initial guess x_1, max iterations maxiter ≥ 2, step size η > 0
# 2. for 2 ≤ i ≤ maxiter:
#   x_i = x_{i-1} + η * f'(x_{i-1})

################################################################"
# f(x) = 0.3cos(3π*x) - x^2
################################################################"
f(x) = 0.3cos(3π*x) - x^2
f_string = "0.3\\cos(3\\pi x) -x^2"
df(x) = ForwardDiff.derivative(f, x)
x_opt = 0.0
y_opt = 0.3
x_subopt1 = -2/3
y_subopt1 = f(x_subopt1)

fig = Figure(resolution = (800, 600))
Label(fig[0, 1:3],
      L"\text{Maximize }f(x)=%$(f_string)\text{ with }\eta = %$η, \text{ }x_{\text{init}} \in \{%$(x_init_a), %$(x_init_b)\}",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)

maxiter = 20
# η = 0.25
# η = 0.2
η = 0.01

x_init_a = -0.8
x_seq_a = gradient_method_1dim(f, x_init_a, η, maxiter)
f_seq_a = f.(x_seq_a)

x_init_b = 0.25
x_seq_b = gradient_method_1dim(f, x_init_b, η, maxiter)
f_seq_b = f.(x_seq_b)

xs = range(-2, 2, length=100)


################################################################
################################################################
# function, optimum, and initial values
################################################################
################################################################

################################################################
# f with x_init_a = -0.8
################################################################
ax = Axis(fig[1, 1],
          title = L"y = %$(f_string)", xlabel = L"x", ylabel = L"y")
lines!(ax, xs, f.(xs), color = :blue, label = "function")
scatter!(ax, [x_opt], [f(x_opt)], color = :red, label = "optimal")
scatter!(ax, [x_init_a], [f(x_init_a)], color = :purple, markersize = 10, label = "initial a")
vlines!(ax, [x_subopt1], label = "≈subopt1")
hlines!(ax, [y_subopt1], label = "≈f(subopt1)")
axislegend(ax; position = :rb,
           # backgroundcolor = :transparent,
           backgroundcolor = (:white, 0.75),
           )

################################################################
# f with x_init_b = 0.25
################################################################
ax = Axis(fig[2, 1],
          title = L"y = %$(f_string)", xlabel = L"x", ylabel = L"y")
lines!(ax, xs, f.(xs), color = :blue, label = "function")
scatter!(ax, [x_opt], [f(x_opt)], color = :red, label = "optimal")
scatter!(ax, [x_init_b], [f(x_init_b)], color = :purple, markersize = 10, label = "initial b")
axislegend(ax; position = :rb,
           # backgroundcolor = :transparent,
           backgroundcolor = (:white, 0.75),
           )


################################################################
################################################################
# optimization process
################################################################
################################################################
xs = range(-1, 1, length=100)

################################################################
# f with x_init_a = -0.8, opt process
################################################################
ax = Axis(fig[1, 2],
          title = L"y = %$(f_string)", xlabel = L"x", ylabel = L"y")
lines!(ax, xs, f.(xs), color = :blue, label = "function")
scatter!(ax, [x_opt], [f(x_opt)], color = :red, label = "optimal")
scatter!(ax, x_seq_a, f_seq_a, color = :green, markersize = 10, label = "initial a")
axislegend(ax; position = :rb,
           # backgroundcolor = :transparent,
           backgroundcolor = (:white, 0.75),
           )

################################################################
# f with x_init_b = 0.25, opt process
################################################################
ax = Axis(fig[2, 2],
          title = L"y = %$(f_string)", xlabel = L"x", ylabel = L"y")
lines!(ax, xs, f.(xs), color = :blue, label = "function")
scatter!(ax, [x_opt], [f(x_opt)], color = :red, label = "optimal")
scatter!(ax, x_seq_b, f_seq_b, color = :green, markersize = 10, label = "initial b")
axislegend(ax; position = :rb,
           # backgroundcolor = :transparent,
           backgroundcolor = (:white, 0.75),
           )

################################################################
# suboptimal
################################################################
ax = Axis(fig[1, 3],
          title = L"\max\text{ }f", xlabel = "iteration", ylabel = L"f")
lines!(ax, 1:maxiter, f_seq_a, color = :green,
       # label = "optimization path",
       )
scatter!(ax, 1:maxiter, f_seq_a, color = :green, marker = :diamond, label = L"f(x)\text{ sequence}")
hlines!(ax, [y_opt], color = :red, label = L"\max\text{ }f")
hlines!(ax, [y_subopt1], label = "≈f(subopt1)")
axislegend(ax; position = :rb,
           # backgroundcolor = :transparent,
           backgroundcolor = (:white, 0.75),
           )

ax = Axis(fig[2, 3],
          title = L"\max\text{ }f", xlabel = "iteration", ylabel = L"f")
lines!(ax, 1:maxiter, f_seq_b, color = :green,
       # label = "optimization path",
       )
scatter!(ax, 1:maxiter, f_seq_b, color = :green, marker = :diamond, label = L"f(x)\text{ sequence}")
hlines!(ax, [y_opt], color = :red, label = L"\max\text{ }f")
axislegend(ax; position = :rb,
           # backgroundcolor = :transparent,
           backgroundcolor = (:white, 0.75),
           )

fig |> display





# ax3 = Axis(fig[2, 1],
#            title = L"\max f", xlabel = L"x", ylabel = L"f(x)")
# lines!(ax3, xs, f.(xs), color = :blue, label = "function")
# scatter!(ax3, x_seq, f_seq, color = :green, marker = :diamond, label = "optimization path")
# scatter!(ax3, [x_opt], [f(x_opt)], color = :red, label = "optimal")
# axislegend(ax3; position = :rb)

# ax4 = Axis(fig[2, 2:3],
#            title = L"x", xlabel = "iteration", ylabel = L"x")
# lines!(ax4, 1:maxiter, x_seq, color = :green, label = L"x\text{ sequence}")
# scatter!(ax4, 1:maxiter, x_seq, marker = :diamond, color = :green,
#          #label = "function"
#          )
# hlines!(ax4, [x_opt], color = :red, linestyle = :dash, label = "optimal x")
# axislegend(ax4; position = :rb)

# fig |> display
# safesave(plotsdir(program_name * "_η=$(η)" * ".pdf"), fig)
