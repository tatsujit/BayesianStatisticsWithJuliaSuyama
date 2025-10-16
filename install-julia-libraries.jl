# install_packages.jl
using Pkg

packages = [
    # データ処理
    "DataFrames",
    "CSV",
    "XLSX",
    "Query",
    "DataFramesMeta",

    # 統計・機械学習
    "Statistics",
    "StatsBase",
    "Distributions",
    "HypothesisTests",
    "GLM",
    "MLJ",

    # 可視化
    "CairoMakie",
    "GLMakie",
    "AlgebraOfGraphics",
    "LaTeXStrings",

    # 数値計算
    "LinearAlgebra",
    "DifferentialEquations",
    "Optimization",
    "ForwardDiff",

    # ユーティリティ
    "ProgressMeter",
    "BenchmarkTools",
    "Revise",
    "DrWatson",
    "Parameters",
    "Chain",
    "IceCream",

    # for REPL
    "OhMyREPL",

    # その他
    "DaemonMode", # for running Julia scripts in the background
]

println("Installing $(length(packages)) packages...")
Pkg.add(packages)
println("Installation complete!")
