using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
using CairoMakie

program_name = "p037-two-variable-func-vector-field"

L = 10 # resolution
xs1 = range(-1, 1, length=L)
xs2 = range(-1, 1, length=L)

f2(x) = -(x .+ 1)' * (x .- 1)

∇f2(x) = -2x

fig = Figure(resolution = (800, 400))
ax = Axis(fig[1, 1],
          title = "f2(x1, x2)", xlabel = "x1", ylabel = "x2")
cf = contourf!(ax, xs1, xs2,
         (x, y) -> f2([x, y]),
         levels = -3:0.25:3,
         colormap = :plasma)
Colorbar(fig[1, 2], cf, label = L"f(x, y)")

ax2 = Axis(fig[1, 3],
           title = "∇f2(x1, x2)", xlabel = "x1", ylabel = "x2")
# grid points
x1 = repeat(xs1, 1, L)
x2 = repeat(xs2', L, 1)
# gradient field
u = [∇f2([x, y])[1] for x in xs1, y in xs2]
v = [∇f2([x, y])[2] for x in xs1, y in xs2]

quiver!(ax2,
        vec(x1), vec(x2),
        vec(u), vec(v),
        lengthscale = 0.1,
        arrowsize = 10,
        arrowcolor = :blue,
        linecolor = :black
        )
fig |> display

safesave(plotsdir(program_name * ".pdf"), fig)
