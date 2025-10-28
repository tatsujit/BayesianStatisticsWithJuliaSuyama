using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p120-Dirichlet-with-contour"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf
using IceCream
"""
これを使うとエラーを抑制できる？
"""
function Base.joinpath(s::String, n::Nothing)
    joinpath(s)
end

include(srcdir("utility_functions.jl"))

fn_suffix = "_contourf_"
# fn_suffix = "_heatmap_"
# fn_suffix = "_contour_"         #
save_flag = true
# save_flag = false
pdf_file_too_big = true # then png
# pdf_file_too_big = false # then png


################################################################
# Beta
################################################################
xs = range(0, 1, length=500)
ys = range(0, 1, length=500)
αs = [[0.75, 0.75, 0.75],
      [0.1, 0.1, 0.1],
      [0.5, 0.5, 0.5],
      [1.0, 1.0, 1.0],
      [2.0, 2.0, 2.0],
      [5.0, 5.0, 5.0],
      [0.1, 0.1, 0.5],
      [0.1, 0.5, 1.0],
      [0.1, 0.5, 5.0],
      [1.0, 2.0, 5.0],
      [10, 20, 50],
      [11, 21, 51],
      ]
# fn_suffix = "_params_below_2.0_"
# fn_suffix = "_params_above_2.0_"

params = αs
n_params = length(params)
n_row, n_col = n2grid(n_params)

################################################################
# plot init
################################################################
fig = Figure(
    size = (1200, 1200),
    figure_padding = 50
)

str1 = "\\text{Dirichlet}(\\alpha_1, \\alpha_2, \\alpha_3)\\text{ distributions. The red circle is the mean.}"
Label(fig[0, 1:n_col],
      LaTeXString(str1),
      fontsize = 16, font = :bold)

################################################################
# multiple Dirichlets
################################################################
hm = nothing
zss = Vector{Matrix{Float64}}(undef, n_params) # zs の条件として、 x+y>1.0だと z==0.0で pdf が Inf になってしまう

################################################################
# prob density calculation
################################################################
for (i, α) in enumerate(params)
    d = Dirichlet(α)
    # x + y > 1.0 だと z==0.0 が許容されるので pdf が Inf になる
    zs = [(x + y >= 1.0 || x <= 0.0 || y <= 0.0) ? 0.0 : pdf(d, [x, y, 1-x-y]) for x in xs, y in ys]
    zss[i] = zs
end
vmin = minimum(vcat(zss...))
vmax = maximum(vcat(zss...))
colorrange = (vmin, vmax)
levels = vcat(0.0:0.1:1.0, 2.0:1.0:10.0)
# levels = range(vmin, vmax, length=10)
# levels = exp.(range(log(vmin), log(vmax), length=10))

################################################################
# check the condifions for Inf
################################################################
# infs = [findall(z -> z==Inf, zss[i]) for i in 1:n_params]
# d = Dirichlet(params[1])
# densities = zss[1]
# inf_pos = findall(isinf, densities) # 56
# x_inf_idx = inf_pos[1][1]
# y_inf_idx = inf_pos[1][2]
# x = xs[x_inf_idx]
# y = ys[y_inf_idx]
# xyz = [x, y, 1-x-y]
# pdf(d, [x, y, 1-x-y])
# pdf(d, [x, y, 1-x-y])
# densities[inf_pos]

################################################################
# plot
################################################################
for (i, α) in enumerate(params)
    # @ic (i, α)
    # row, col = n2ij(i, n_params; n_row = n_row, n_col = n_col)
    row, col = n2ij(i, n_params)
    d = Dirichlet(α)
    μ = mean(d)
    # @ic μ
    X = rand(d, 10000)

    ax1 = Axis(fig[row, col],
               title = "Dirichlet($α)", #", P(μ±σ)=$(oneσpf)",
               xlabel = L"x",
               ylabel = L"y",
               # aspect = DataAspect(),
               aspect = 1,
               # limits = ((0, μ+3σ), nothing),
               limits = ((0, 1), (0, 1)),
               )
    scatter!(ax1, X, color = (:blue, 0.1))
    scatter!(ax1, [μ[1]], [μ[2]], color = (:red, 1.0), markersize = 20)

    ax2 = Axis(fig[row+n_row, col],
               title = "Dirichlet($α)", #", P(μ±σ)=$(oneσpf)",
               xlabel = L"x",
               ylabel = L"y",
               # aspect = DataAspect(),
               aspect = 1,
               # limits = ((0, μ+3σ), nothing),
               limits = ((0, 1), (0, 1)),
               )

    if fn_suffix == "_heatmap_"
        # global hm = heatmap!(ax, xs, ys, zss[i], colorrange = colorrange)
        global hm = heatmap!(ax2, xs, ys, zss[i],
                             colorrange = (0, 3),
                             )
    elseif fn_suffix == "_contourf_"
        global hm = contourf!(ax2, xs, ys, zss[i],
                              # levels = levels,
                              # nan_color=:transparent,
                              # linewidth = 1,
                              # labels = true,
                              )
    elseif fn_suffix == "_contour_"
        global hm = contour!(ax2, xs, ys, zss[i],
                             levels = levels,
                             nan_color=:transparent,
                             linewidth = 1,
                             labels = true,
                             colormap = :viridis
                             )
    end
end

if fn_suffix ∈ ["_heatmap_", "_contourf_"]
    Colorbar(fig[n_row+1:2n_row, n_col+1], hm, label = "probability",
             colorrange = (0, 3),
             # colormap = :viridis
             )
end

fig |> display
if save_flag
    if pdf_file_too_big
        safesave(plotsdir(program_name * fn_suffix * ".png"), fig)
    else
        safesave(plotsdir(program_name * fn_suffix * ".pdf"), fig)
    end
end
