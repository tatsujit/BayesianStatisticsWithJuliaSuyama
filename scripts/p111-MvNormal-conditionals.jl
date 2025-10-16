using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p109-MvNormal-marginals"
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

################################################################
# Multivariate Normal
################################################################
μs = [[0.0, 0.0],
      [0.0, 0.0],
      [0.0, 0.0],
      [0.0, 0.0],
      [0.0, 0.0],
      ]
Σs = [
    [1.0 0.0;
     0.0 1.0],
    [1.0 0.5;
     0.5 1.0],
    [1.0 -0.5;
     -0.5 1.0],
    [1.0 0.9;
     0.9 1.0],
    [1.5 0.25;
     0.25 1.5],

    # LoadError: PosDefException: matrix is not positive definite; Factorization failed.
    # [1.0 1.0;
    #  1.0 1.0],
    # LoadError: PosDefException: matrix is not Hermitian; Factorization failed.
    # [1.0 0.5;
    #  -0.5 1.0],
]
@assert length(μs) == length(Σs)
n_row, n_col = n2grid(length(μs))

################################################################
# plot init
################################################################
fig = Figure(
    size = (800, 800),
    figure_padding = 30
)
Label(fig[0, 1:2],
      "Multivariate Normal distributions", #"\mu=%$μ\text{, }\sigma=%$σ",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)

x1_lower, x1_upper = -5, 5
x2_lower, x2_upper = x1_lower, x1_upper

x1s = range(x1_lower, x1_upper, length=100)
x2s = range(x2_lower, x2_upper, length=100)
limits = ((x1_lower, x1_upper), (x2_lower, x2_upper))

################################################################
# MvNormal and the marginals
################################################################

i = 5
d = MvNormal(μs[i], Σs[i])

################################################################
# p(x_1, x_2)
################################################################
row, col = 1, 1
ax1 = Axis(fig[row, col],
          xlabel = L"x_1",
          ylabel = L"x_2",
          title = L"p(x_1, x_2)",
          limits = limits,
          )

contour!(ax1, x1s, x2s, (x1, x2) -> pdf(d, [x1, x2]),
         levels = 6,
         linewidth = 1,
         labels = true,
         )
text!(ax1, 0.5, 0.9,  # (0.5, 0.9) = 中央上部
      text = "μ = $(μs[i]), Σ = $(Σs[i])",
      align = (:center, :top),
      space = :relative,  # これが重要
      fontsize = 14)

################################################################
# p(x_2 | x_1)
################################################################
row, col = 1, 2
ax2 = Axis(fig[row, col],
          xlabel = L"\text{density}",
          ylabel = L"x_2",
          title = "cond. prob. density",
          limits = ((0, 0.5), (x2_lower, x2_upper)),
          )

x2_range = range(x2_lower, x2_upper, length=100)
px1_marginal(x1) = approx_integration(x2_range, x2 -> pdf(d, [x1, x2]))[1]

x1 = 2.0
px2_conditional(x2) = pdf(d, [x1, x2]) / px1_marginal(x1)
densities = px2_conditional.(x2_range)
lines!(ax2, densities, x2_range, label = L"p(x_2|x_1 = %$(x1))", color = :orange)
vlines!(ax1, [x1], color = :orange, linestyle = :dash)

x1 = -2.0
densities = px2_conditional.(x2_range)
lines!(ax2, densities, x2_range, label = L"p(x_2|x_1 = %$(x1))", color = :green)
vlines!(ax1, [x1], color = :green, linestyle = :dash)

axislegend(ax2)

################################################################
# p(x_1 | x_2)
################################################################
row, col = 2, 1
ax3 = Axis(fig[row, col],
           xlabel = L"x_1",
           ylabel = L"\text{density}",
           title = "cond. prob. density",
           limits = ((x1_lower, x1_upper), (0, 0.5)),
           )

x1_range = range(x1_lower, x1_upper, length=100)
px2_marginal(x2) = approx_integration(x1_range, x1 -> pdf(d, [x1, x2]))[1]

x2 = 0.0
px1_conditional(x1) = pdf(d, [x1, x2]) / px2_marginal(x2)
densities = px1_conditional.(x1_range)
lines!(ax3, x1_range, densities, label = L"p(x_1|x_2 = %$(x2))", color = :red)
hlines!(ax1, [x2], color = :red, linestyle = :dash)

x2 = 1.0
densities = px1_conditional.(x1_range)
lines!(ax3, x1_range, densities, label = L"p(x_1|x_2 = %$(x2))", color = :blue)
hlines!(ax1, [x2], color = :blue, linestyle = :dash)

axislegend(ax3)

################################################################
# display and save
################################################################
fig |> display
safesave(plotsdir(program_name * ".pdf"), fig)
