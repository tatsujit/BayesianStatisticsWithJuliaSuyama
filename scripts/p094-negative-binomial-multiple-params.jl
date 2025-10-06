using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p094-negative-binomial-multiple-params"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf

################################################################
# NegativeBinomial
################################################################
xs = 0:60
rs = [3, 5, 10]
μs = [0.3, 0.5, 0.7]
d = NegativeBinomial(r, μ)
n_rs = length(rs)
n_μs = length(μs)
n_params = n_rs * n_μs
n_row, n_col = n2grid(n_params)

################################################################
# plot init
################################################################
fig = Figure(
    size = (1200, 800),
    figure_padding = 30
)
Label(fig[0, 1:n_col],
      L"\text{negative Binomial distributions; mean = }\frac{r(1-μ)}{μ}\text{, var = }\frac{r(1-μ)}{μ^2}",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)
################################################################
# multiple Poissons
################################################################
for (i, (r, μ)) in enumerate(Iterators.product(rs, μs))
    row, col = n2ij(i, n_params)
    d = NegativeBinomial(r, μ)
    m = mean(d)
    v = var(d)
    σ = sqrt(v)
    m_str = @sprintf("%.2f", m)
    v_str = @sprintf("%.2f", v)
    σ_str = @sprintf("%.2f", σ)
    # [μ-σ, μ+σ] の面積
    # sm, sp = cdf(d, [μ-σ, μ+σ])
    # oneσp = sp-sm
    # oneσpf = @sprintf("%.2f", oneσp)

    ax = Axis(fig[row, col],
              title = "NegativeBinomial(r=$r, μ=$μ)", #", P(μ-σ)-P(μ+σ)=$(oneσpf)",
              xlabel = L"x",
              ylabel = "probability",
              # aspect = DataAspect(),
              aspect = 2,
              )
    barplot!(ax, xs, pdf.(d, xs))
    text!(ax, 0.7, 0.9,  # (0.5, 0.9) = 中央上部
          text = "mean = $(m_str), var = $(v_str)\n std = $(σ_str)",
          align = (:center, :top),
          space = :relative,  # これが重要
          fontsize = 14)
    lw = 2
    vlines!(ax, [m], label="mean", color = :red, linewidth = lw)
    vlines!(ax, [m-σ, m+σ], label="mean±σ", color = :orange, linestyle = :dash, linewidth = lw)
end

fig |> display
safesave(plotsdir(program_name * ".pdf"), fig)
