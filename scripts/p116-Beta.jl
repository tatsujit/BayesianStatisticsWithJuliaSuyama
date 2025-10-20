using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p116-Beta"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf

################################################################
# Beta
################################################################
xs = range(0, 1, length=100)
# αs = [0.1, 0.3, 1.0, 5.0, 100.0]
# βs = [0.2, 0.6, 1.0, 2.0, 5.0, 10.0, 100.0, 200.0]

# fn_suffix = "_params_below_2.0_"
# αs = [0.1, 0.3, 1.0, 2.0]
# βs = [0.2, 0.6, 1.0, 2.0]
fn_suffix = "_params_above_2.0_"
αs = [1.0, 5.0, 100.0]
βs = [2.0, 5.0, 10.0, 100.0, 200.0]

params = vec(collect(Iterators.product(αs, βs)))

μs = [α / (α + β) for (α, β) in params]
νs = [α + β for (α, β) in params]
modes = [(α - 1) / (α + β - 2) for (α, β) in params]
vars = [α * β / ((α + β)^2 * (α + β + 1)) for (α, β) in params]
n_params = length(params)
# n_row, n_col = n2grid(n_params)
n_row, n_col = length(βs), length(αs)

################################################################
# plot init
################################################################
fig = Figure(
    # size = (2400, 1600),
    size = (1200, 1200),
    # size = (1200, 800),
    figure_padding = 50
)

str1 = "\\text{Beta}(\\alpha, \\beta)\\text{ distributions; }"
str2 = "\\alpha-1, \\beta-1 \\text{ are the number of successes and failures. }"
str3 = "\\text{mean, precision, mode, var are, respectively, }"
str4 = "\\mu, \\nu, \\text{mode}, \\text{ var}."
Label(fig[0, 1:n_col],
      # L"\text{Gamma}(\alpha, \theta)\text{ distributions; }\mu = \alpha\theta\text{, }\sigma^2 = \alpha\theta^2",
      LaTeXString(str1 * str2 * str3 * str4),
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)

################################################################
# multiple Poissons
################################################################
for (i, (α, β)) in enumerate(params)

    row, col = n2ij(i, n_params; n_row = n_row, n_col = n_col)
    d = Beta(α, β)

    μ = μs[i]
    σ² = vars[i]
    σ = sqrt(σ²)
    ν = νs[i]
    mode = modes[i]

    μ_str = @sprintf("%.2f", μ)
    σ²_str = @sprintf("%.2f", σ²)
    σ_str = @sprintf("%.2f", σ)
    ν_str = @sprintf("%.2f", ν)
    mode_str = @sprintf("%.2f", mode)

    # [μ-σ, μ+σ] の面積
    sm, sp = cdf(d, [μ-σ, μ+σ])
    oneσp = sp-sm
    oneσpf = @sprintf("%.2f", oneσp)

    ax = Axis(fig[row, col],
              title = "Beta(α=$α, β=$β), P(μ±σ)=$(oneσpf)",
              xlabel = L"x",
              ylabel = "prob. density",
              # aspect = DataAspect(),
              aspect = 2,
              # limits = ((0, μ+3σ), nothing),
              limits = ((0, 1), nothing),
              )
    lw = 2
    vlines!(ax, [μ], label="mean", color = :red, linewidth = lw)
    vlines!(ax, [μ-σ, μ+σ], label="mean±σ", color = :orange, linestyle = :dash, linewidth = lw)
    lines!(ax, xs, pdf.(d, xs))
    text!(ax, 0.5, 0.9,  # (0.5, 0.9) = 中央上部
          text = "μ = $(μ_str), σ² = $(σ²_str)\n σ = $(σ_str)",
          align = (:center, :top),
          space = :relative,  # これが重要
          fontsize = 14)
end

fig |> display
safesave(plotsdir(program_name * fn_suffix * ".pdf"), fig)
