# 尝试直接下载（使用 dl=1）
curl -L --progress-bar -o datasets.zip 'https://www.dropbox.com/s/8n02xqv3l9q18r1/datasets.zip?dl=1' || \
# 如果失败，尝试使用 dl=0 并跟随重定向
curl -L --progress-bar -o datasets.zip 'https://www.dropbox.com/s/8n02xqv3l9q18r1/datasets.zip?dl=0'

if [ -f datasets.zip ]; then
    echo "下载完成，正在解压..."
    unzip -q datasets.zip
    rm -rf datasets.zip
    echo "数据下载并解压完成！"
else
    echo "错误：下载失败，请检查网络连接或手动下载数据集"
    exit 1
fi
