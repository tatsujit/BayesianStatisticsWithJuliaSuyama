using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p101-normal-cdf-and-areas"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf
using IceCream

include(srcdir("utility_functions.jl"))

################################################################
# NegativeBinomial
################################################################
μ = 0.0
σ = 0.2
d = Normal(μ, σ)
ranges = repeat([(-Inf, 0.0), (-Inf, 0.2), (0.0, 0.2)], 2) # pdfs and cdfs
n_params = length(ranges)
n_row, n_col = n2grid(n_params)

################################################################
# plot init
################################################################
fig = Figure(
    size = (1200, 800),
    figure_padding = 30
)
Label(fig[0, 1:n_col],
      L"\text{Normal distribution}\mu=%$μ\text{, }\sigma=%$σ",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)
################################################################
# multiple Normals
################################################################
for (i, (a, b)) in enumerate(ranges)
    @ic i
    row, col = n2ij(i, n_params)
    m = mean(d)
    v = var(d)
    σ = sqrt(v)
    m_str = @sprintf("%.2f", m)
    v_str = @sprintf("%.2f", v)
    σ_str = @sprintf("%.2f", σ)
    # [μ-σ, μ+σ] の面積
    sm, sp = cdf(d, [μ-σ, μ+σ])
    oneσp = sp-sm
    oneσpf = @sprintf("%.2f", oneσp)

    ax = Axis(fig[row, col],
              title = "Normal(μ=$μ, σ=$σ), [$a, $b]",
              xlabel = L"x",
              ylabel = "probability",
              # aspect = DataAspect(),
              aspect = 2,
              )

    xs = -1:0.01:1

    if i <= div(n_params, 2) # pdfs
        lines!(ax, xs, pdf.(d, xs))
        text!(ax, 0.7, 0.9,  # (0.5, 0.9) = 中央上部
              text = "mean = $(m_str), var = $(v_str)\n std = $(σ_str), ±σ = $oneσpf",
              align = (:center, :top),
              space = :relative,  # これが重要
              fontsize = lw)
        14 = 2
        vlines!(ax, [m], label="mean", color = :red, linewidth = lw)
        vlines!(ax, [m-σ, m+σ], label="mean±σ", color = :orange, linestyle = :dash, linewidth = lw)

        # 色をつける範囲 (例: -1 ≤ x ≤ 1)
        a = a == -Inf ? -1.0 : a
        x_fill = range(a, b, length=200)
        y_fill = pdf.(d, x_fill)

        # 範囲に色をつける
        band!(ax,
              x_fill, # x
              fill(0, length(x_fill)), y_fill, # y_lower, y_upper
              color=(:blue, 0.3),
              label="P($a ≤ X ≤ $b)")
    else # cdfs
        lines!(ax, xs, cdf.(d, xs))
        lw = 2
        vlines!(ax, [m], label="mean", color = :red, linewidth = lw)
        vlines!(ax, [m-σ, m+σ], label="mean±σ", color = :orange, linestyle = :dash, linewidth = lw)

        lw2 = 2
        if a != -Inf
            lines!(ax, [a, a], [0.0, cdf(d, a)],
                   color=(:blue, 0.9),
                   label="P($a ≤ X ≤ $b)",
                   # linestyle = :dash,
                   linewidth = lw2,
                   )
        end
        lines!(ax, [b, b], [0.0, cdf(d, b)],
               color=(:blue, 0.9),
               label="P($a ≤ X ≤ $b)",
               # linestyle = :dash,
               linewidth = lw2,
               )
    end
end

fig |> display
safesave(plotsdir(program_name * ".pdf"), fig)
