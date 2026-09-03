# Face-payment 静态展示

这些页面是仓库的 **GitHub Pages 架构 / 结果说明**，不是可以在浏览器里跑的人脸支付演示。

- 计划地址：https://unstoppablecurry.github.io/Face-payment/
- 发布方式：仓库根目录 `.github/workflows/static.yml`（GitHub 官方 static Pages 工作流，上传 `docs/`）
- 允许使用的素材：README 架构文字、`ArcFace/loss.png`、README 首张 user-images 配图、Bilibili `BV1bL4y1s7Fr`

本地预览：

```bash
python3 -m http.server 8080 --directory docs
```

然后打开 http://127.0.0.1:8080/ 。项目站点上线后，资源路径仍是相对路径，因此 `/Face-payment/` 前缀下同样有效。
