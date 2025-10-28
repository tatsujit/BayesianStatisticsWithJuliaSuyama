using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p125-6-Gaussian-mixture"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf
using IceCream

include(srcdir("utility_functions.jl"))

################################################################
# Multivariate Normal
################################################################
ps = [0.2, 0.5, 0.8] # latent
μs = [[-1.0, 1.0],
      [1.0, -1.0],
      ]
Σs = [
    [0.2 0.1;
     0.1 0.2],
    [0.4 -0.1;
     -0.1 0.4],
    # [1.0 0.0;
    #  0.0 1.0],
    # [1.0 0.5;
    #  0.5 1.0],
    # [1.0 -0.5;
    #  -0.5 1.0],
    # [1.0 0.9;
    #  0.9 1.0],
]
@assert length(μs) == length(Σs)
n_row, n_col = n2grid(length(ps)) # or 1, 3

################################################################
# plot init
################################################################
fig = Figure(
    size = (1200, 1200),
    figure_padding = 30
)
Label(fig[0, 1:n_col],
      "Mixture Gaussian distributions", #"\mu=%$μ\text{, }\sigma=%$σ",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)

x_lower, x_upper = -3, 3
y_lower, y_upper = x_lower, x_upper

xs = range(x_lower, x_upper, length=100)
ys = range(y_lower, y_upper, length=100)
limits = ((x_lower, x_upper), (y_lower, y_upper))

################################################################
# multiple mixtures
################################################################
hm = nothing
for (i, p) in enumerate(ps)
    d1 = MvNormal(μs[1], Σs[1])
    d2 = MvNormal(μs[2], Σs[2])
    row, col = n2ij(i, length(ps))

    ax1 = Axis(fig[row, col],
              xlabel = L"x",
              ylabel = L"y",
              title = "prob. density, p = ($p, $(round(1-p, digits=1)))",
              limits = limits,
              )

    contour!(ax1, xs, ys, (x, y) -> ((1-p) * pdf(d1, [x, y]) + p * pdf(d2, [x, y])),
             levels = 10,
             linewidth = 1,
             labels = true,
             )

    ax2 = Axis(fig[row + n_row, col],
              xlabel = L"x",
              ylabel = L"y",
              title = "prob. density",
              limits = limits,
              )

    global hm = contourf!(ax2, xs, ys, (x, y) -> ((1-p) * pdf(d1, [x, y]) + p * pdf(d2, [x, y])),
             levels = 10,
             )
    # text!(ax, 0.5, 0.9,  # (0.5, 0.9) = 中央上部
    #       text = "μ = $(μs[i]), Σ = $(Σs[i])",
    #       align = (:center, :top),
    #       space = :relative,  # これが重要
    #       fontsize = 14)

    ################################################################
    # data sampling
    ################################################################
    N = 1000
    X = Array{Float64}(undef, 2, N)
    S = Array{Int}(undef, N)

    for i in 1:N
        s = 1 + (rand(Bernoulli(p)) ? 1 : 0)
        S[i] = s
        (μ, Σ) = (μs[s], Σs[s])
        X[:, i] = rand(MvNormal(μ, Σ))
    end

    ax3 = Axis(fig[row + 2n_row, col],
              xlabel = L"x",
              ylabel = L"y",
              title = "sample (N=$N)",
              limits = limits,
              )

    scatter!(ax3, X,
             color = (:blue, 0.3)
             )

end
Colorbar(fig[2,4], hm)

fig |> display
safesave(plotsdir(program_name * ".pdf"), fig)
