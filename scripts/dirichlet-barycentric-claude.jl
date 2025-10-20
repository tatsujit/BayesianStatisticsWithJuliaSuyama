# LoadError: MethodError: no method matching bounds(::Polygon{2, Float32, Point{2, Float32}, LineString{2, Float32, Point{2, Float32}, Base.ReinterpretArray{Line{2, Float32}, 1, Tuple{Point{2, Float32}, Point{2, Float32}}, TupleView{Tuple{Point{2, Float32}, Point{2, Float32}}, 2, 1, Vector{Point{2, Float32}}}, false}}, Vector{LineString{2, Float32, Point{2, Float32}, Base.ReinterpretArray{Line{2, Float32}, 1, Tuple{Point{2, Float32}, Point{2, Float32}}, TupleView{Tuple{Point{2, Float32}, Point{2, Float32}}, 2, 1, Vector{Point{2, Float32}}}, false}}}})
# Closest candidates are:
#   bounds(!Matched::Extents.Extent)
#    @ Extents ~/.julia/packages/Extents/XER9E/src/Extents.jl:60
# in expression starting at /Users/tatsujit/git/BayesianStatisticsWithJuliaSuyama/scripts/dirichlet-barycentric-claude.jl:188

using CairoMakie
using Distributions
using GeometryOps
using GeoInterface
using GeometryBasics

"""
    create_simplex_triangle()

シンプレックスの三角形ジオメトリを作成
"""
function create_simplex_triangle()
    # 三角形の頂点
    points = Point2f[
        (0.0, 0.0),      # α₁頂点（左下）
        (1.0, 0.0),      # α₂頂点（右下）
        (0.5, sqrt(3)/2) # α₃頂点（上）
    ]
    return Polygon(points)
end

"""
    barycentric_to_cartesian(λ)

重心座標をデカルト座標に変換
"""
function barycentric_to_cartesian(λ)
    vertices = [
        Point2f(0.0, 0.0),      # α₁
        Point2f(1.0, 0.0),      # α₂
        Point2f(0.5, sqrt(3)/2) # α₃
    ]

    x = λ[1] * vertices[1][1] + λ[2] * vertices[2][1] + λ[3] * vertices[3][1]
    y = λ[1] * vertices[1][2] + λ[2] * vertices[2][2] + λ[3] * vertices[3][2]

    return Point2f(x, y)
end

"""
    cartesian_to_barycentric(p::Point2f)

デカルト座標を重心座標に変換
"""
function cartesian_to_barycentric(p::Point2f)
    x, y = p[1], p[2]

    λ3 = 2 * y / sqrt(3)
    λ2 = x - 0.5 * λ3
    λ1 = 1 - λ2 - λ3

    return [λ1, λ2, λ3]
end

"""
    is_inside_simplex(p::Point2f, triangle::Polygon)

点が三角形の内部にあるかをGeometryOpsで判定
"""
function is_inside_simplex(p::Point2f, triangle::Polygon)
    return GeometryOps.contains(triangle, p)
end

"""
    plot_dirichlet_with_geometryops(ax, α; resolution=300)

GeometryOpsを使用したディリクレ分布の描画
"""
function plot_dirichlet_with_geometryops(ax, α; resolution=300)
    dist = Dirichlet(α)
    triangle = create_simplex_triangle()

    # バウンディングボックスを取得
    bbox = GeometryOps.bounds(triangle)
    x_min, x_max = bbox[1][1], bbox[2][1]
    y_min, y_max = bbox[1][2], bbox[2][2]

    # グリッドを少し広めに設定
    margin = 0.1
    x_range = range(x_min - margin, x_max + margin, length=resolution)
    y_range = range(y_min - margin, y_max + margin, length=resolution)

    Z = zeros(resolution, resolution)

    for (i, x) in enumerate(x_range)
        for (j, y) in enumerate(y_range)
            p = Point2f(x, y)

            if is_inside_simplex(p, triangle)
                λ = cartesian_to_barycentric(p)

                # 数値的に安定な範囲でのみ計算
                if all(λ .> 1e-8) && all(λ .< 1 - 1e-8)
                    Z[i, j] = pdf(dist, λ)
                else
                    Z[i, j] = 0.0
                end
            else
                Z[i, j] = NaN
            end
        end
    end

    # 等高線塗りつぶし
    cf = contourf!(ax, x_range, y_range, Z',
                   levels=20,
                   colormap=:Blues,
                   nan_color=:transparent)

    # 等高線
    contour!(ax, x_range, y_range, Z',
             levels=15,
             color=:white,
             linewidth=0.5,
             alpha=0.6)

    return cf
end

"""
    add_simplex_grid!(ax; n_lines=9)

三角座標系のグリッド線を追加
"""
function add_simplex_grid!(ax; n_lines=9)
    for i in 1:n_lines
        t = i/(n_lines + 1)

        # α₁一定の線
        p1 = barycentric_to_cartesian([t, 1-t, 0])
        p2 = barycentric_to_cartesian([t, 0, 1-t])
        lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]],
               color=(:gray, 0.2), linewidth=0.3)

        # α₂一定の線
        p1 = barycentric_to_cartesian([1-t, t, 0])
        p2 = barycentric_to_cartesian([0, t, 1-t])
        lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]],
               color=(:gray, 0.2), linewidth=0.3)

        # α₃一定の線
        p1 = barycentric_to_cartesian([1-t, 0, t])
        p2 = barycentric_to_cartesian([0, 1-t, t])
        lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]],
               color=(:gray, 0.2), linewidth=0.3)
    end
end

"""
    add_simplex_ticks!(ax; n_ticks=5)

三角形の辺に沿った目盛りを追加
"""
function add_simplex_ticks!(ax; n_ticks=5)
    for i in 1:n_ticks-1
        t = i/n_ticks

        # 底辺（α₃ = 0）
        p = barycentric_to_cartesian([1-t, t, 0])
        text!(ax, p[1], p[2] - 0.04,
              text=string(round(t, digits=1)),
              fontsize=8, color=:gray)

        # 左辺（α₂ = 0）
        p = barycentric_to_cartesian([1-t, 0, t])
        text!(ax, p[1] - 0.04, p[2],
              text=string(round(t, digits=1)),
              fontsize=8, color=:gray)

        # 右辺（α₁ = 0）
        p = barycentric_to_cartesian([0, 1-t, t])
        text!(ax, p[1] + 0.03, p[2],
              text=string(round(t, digits=1)),
              fontsize=8, color=:gray)
    end
end

# メイン図の作成
fig = Figure(size=(900, 900))

params = [
    [1.5, 1.5, 1.5],   # 対称、一様に近い
    [5.0, 5.0, 5.0],   # 対称、中心に集中
    [1.0, 5.0, 1.0],   # α₂（右下）に偏り
    [10.0, 1.0, 1.0]   # α₁（左下）に偏り
]

for (idx, α) in enumerate(params)
    row = div(idx-1, 2) + 1
    col = rem(idx-1, 2) + 1

    ax = Axis(fig[row, col],
              aspect=DataAspect(),
              title="α = ($(α[1]), $(α[2]), $(α[3]))")

    # ディリクレ分布のプロット
    cf = plot_dirichlet_with_geometryops(ax, α, resolution=400)

    # グリッド線を追加
    add_simplex_grid!(ax, n_lines=9)

    # 三角形の境界を描画
    triangle = create_simplex_triangle()
    vertices = GeoInterface.coordinates(triangle)[1]
    lines!(ax, vertices, color=:black, linewidth=2.5)

    # 頂点ラベル
    text!(ax, -0.08, -0.06, text="α₁", fontsize=14, font="Helvetica Bold")
    text!(ax, 1.05, -0.06, text="α₂", fontsize=14, font="Helvetica Bold")
    text!(ax, 0.5, sqrt(3)/2 + 0.05, text="α₃", fontsize=14, font="Helvetica Bold")

    # 目盛りを追加（オプション）
    if idx == 1  # 最初のプロットのみ
        add_simplex_ticks!(ax, n_ticks=5)
    end

    hidedecorations!(ax)
    hidespines!(ax)

    xlims!(ax, -0.15, 1.15)
    ylims!(ax, -0.1, sqrt(3)/2 + 0.1)
end

# カラーバー
Colorbar(fig[:, 3],
         limits=(0, 15),
         label="Density",
         colormap=:Blues,
         width=20)

fig |> display
