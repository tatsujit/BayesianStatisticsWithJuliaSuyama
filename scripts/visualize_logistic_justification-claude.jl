using CairoMakie
using Distributions

function visualize_logistic_justification()
    fig = Figure(size=(1200, 800))

    # 1. シグモイド関数とその微分
    ax1 = Axis(fig[1, 1],
               xlabel="z",
               ylabel="σ(z)",
               title="Sigmoid Function and its Derivative")

    z = range(-6, 6, length=500)
    σ(z) = 1 / (1 + exp(-z))
    σ_vals = σ.(z)
    σ_prime = σ_vals .* (1 .- σ_vals)

    lines!(ax1, z, σ_vals, label="σ(z)", linewidth=2)
    lines!(ax1, z, σ_prime, label="σ'(z) = σ(1-σ)",
           linewidth=2, linestyle=:dash)
    axislegend(ax1, position=:lt)

    # 2. 対数オッズとの関係
    ax2 = Axis(fig[1, 2],
               xlabel="log-odds",
               ylabel="Probability p",
               title="Log-odds to Probability")

    logodds = range(-4, 4, length=500)
    p_vals = σ.(logodds)

    lines!(ax2, logodds, p_vals, linewidth=2, color=:blue)
    hlines!(ax2, [0.5], color=:gray, linestyle=:dash)
    vlines!(ax2, [0], color=:gray, linestyle=:dash)

    # 3. ロジスティック分布 vs ガウス分布
    ax3 = Axis(fig[2, 1],
               xlabel="ε",
               ylabel="Density",
               title="Logistic vs Normal Distribution")

    ε = range(-6, 6, length=500)
    logistic_dist = Logistic(0, 1)
    normal_dist = Normal(0, π/sqrt(3))  # 同じ分散

    lines!(ax3, ε, pdf.(logistic_dist, ε),
           label="Logistic(0,1)", linewidth=2)
    lines!(ax3, ε, pdf.(normal_dist, ε),
           label="Normal(0,π/√3)", linewidth=2, linestyle=:dash)
    axislegend(ax3, position=:rt)

    # 4. 対称性の可視化
    ax4 = Axis(fig[2, 2],
               xlabel="z",
               ylabel="Value",
               title="Symmetry: σ(-z) = 1 - σ(z)")

    z_sym = range(-4, 4, length=500)
    σ_pos = σ.(z_sym)
    σ_neg = σ.(-z_sym)

    lines!(ax4, z_sym, σ_pos, label="σ(z)", linewidth=2)
    lines!(ax4, z_sym, σ_neg, label="σ(-z)", linewidth=2, linestyle=:dash)
    lines!(ax4, z_sym, 1 .- σ_pos, label="1 - σ(z)",
           linewidth=2, linestyle=:dot, color=:red)
    axislegend(ax4, position=:lt)

    return fig
end

visualize_logistic_justification() |> display
