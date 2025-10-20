using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p120-Dirichlet"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf

include(srcdir("utility_functions.jl"))

################################################################
# Beta
################################################################
xs = range(0, 1, length=100)
ys = range(0, 1, length=100)
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
# μs = [α / (α + β) for (α, β) in params]
# νs = [α + β for (α, β) in params]
# modes = [(α - 1) / (α + β - 2) for (α, β) in params]
# vars = [α * β / ((α + β)^2 * (α + β + 1)) for (α, β) in params]
n_params = length(params)
n_row, n_col = n2grid(n_params)

################################################################
# plot init
################################################################
fig = Figure(
    # size = (2400, 1600),
    size = (1200, 1200),
    # size = (1200, 800),
    figure_padding = 50
)

str1 = "\\text{Dirichlet}(\\alpha_1, \\alpha_2, \\alpha_3)\\text{ distributions. The red circle is the mean.}"
# str2 = "\\alpha-1, \\beta-1 \\text{ are the number of successes and failures. }"
# str3 = "\\text{mean, precision, mode, var are, respectively, }"
# str4 = "\\mu, \\nu, \\text{mode}, \\text{ var}."
Label(fig[0, 1:n_col],
      # L"\text{Gamma}(\alpha, \theta)\text{ distributions; }\mu = \alpha\theta\text{, }\sigma^2 = \alpha\theta^2",
      LaTeXString(str1), # * str2 * str3 * str4),
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)

################################################################
# multiple Dirichlets
################################################################
for (i, α) in enumerate(params)
    # @ic (i, α)
    # row, col = n2ij(i, n_params; n_row = n_row, n_col = n_col)
    row, col = n2ij(i, n_params)
    d = Dirichlet(α)

    μ = mean(d)
    @ic μ
    # med = median(d)
    # mode = mode(d)
    # μ = μs[i]
    # σ² = vars[i]
    # σ = sqrt(σ²)
    # ν = νs[i]
    # mode = modes[i]

    # μ_str = @sprintf("%.2f", μ)
    # σ²_str = @sprintf("%.2f", σ²)
    # σ_str = @sprintf("%.2f", σ)
    # ν_str = @sprintf("%.2f", ν)
    # mode_str = @sprintf("%.2f", mode)

    # # [μ-σ, μ+σ] の面積
    # sm, sp = cdf(d, [μ-σ, μ+σ])
    # oneσp = sp-sm
    # oneσpf = @sprintf("%.2f", oneσp)

    ax = Axis(fig[row, col],
              title = "Dirichlet($α)", #", P(μ±σ)=$(oneσpf)",
              xlabel = L"x",
              ylabel = L"y",
              # aspect = DataAspect(),
              aspect = 1,
              # limits = ((0, μ+3σ), nothing),
              limits = ((0, 1), (0, 1)),
              )
    lw = 2
    # vlines!(ax, [μ], label="mean", color = :red, linewidth = lw)
    # vlines!(ax, [μ-σ, μ+σ], label="mean±σ", color = :orange, linestyle = :dash, linewidth = lw)
    # lines!(ax, xs, pdf.(d, xs))

    X = rand(d, 10000)
    scatter!(ax, X, color = (:blue, 0.1))
    scatter!(ax, [μ[1]], [μ[2]], color = (:red, 1.0), markersize = 20)
    # text!(ax, 0.5, 0.9,  # (0.5, 0.9) = 中央上部
    #       text = "μ = $(μ_str), σ² = $(σ²_str)\n σ = $(σ_str)",
    #       align = (:center, :top),
    #       space = :relative,  # これが重要
    #       fontsize = 14)
end

fig |> display
# safesave(plotsdir(program_name * ".pdf"), fig)
