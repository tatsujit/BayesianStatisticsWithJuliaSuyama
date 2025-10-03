"""
    sign_changes(v::AbstractVector{<:Real}) -> Int

実数ベクトルの符号切り替わり回数を返す。
ゼロは無視される。
"""
function sign_changes(v::AbstractVector{<:Real})
    # ゼロでない要素の符号のみを抽出
    signs = sign.(filter(!iszero, v))
    # 符号が変わった回数をカウント
    return count(i -> signs[i] != signs[i+1], 1:length(signs)-1)
end
