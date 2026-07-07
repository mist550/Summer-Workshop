# 2.9 转发编译包

本目录包含异地编译所需的最小文件：

- `2.9.tex`：LaTeX 源文件
- `2.9.pdf`：已编译好的 PDF
- `figures/`：正文引用的 2 张图片

如需重新编译，请在本目录下运行：

```powershell
latexmk -xelatex -interaction=nonstopmode -file-line-error 2.9.tex
```

不需要原始代码、数据文件或本地实验目录。
