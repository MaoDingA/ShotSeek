# M0 黄金样片

M0 使用 Blender Foundation 开放电影《Tears of Steel》的 75 秒片段验证视频理解、ASR 和统一时间线契约。

- 原片：`Tears of Steel`，Blender Foundation / Blender Institute
- 官方页面：https://studio.blender.org/films/tears-of-steel/
- 官方下载：https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov
- 许可证：Creative Commons Attribution 3.0（影片内容）；使用时保留原作者署名
- 截取区间：原片 `00:04:52.000` 至 `00:06:07.000`
- 本地输出：`samples/golden.mp4`，75 秒、720p、H.264/AAC mono
- 视频大小：21,782,705 bytes
- 视频 SHA256：`9a11b716f750bd61f081c47f2195ca3fdacf8b098891d862c273bfd172c50aa8`
- 音频 SHA256：`2bb2a3f394a36d6652e4e2d33a0558e37c3c6f0197d1524bbcb144c66debf6cb`

真实视频和音频文件由 `.gitignore` 排除，不进入公开仓库。生成后请运行：

```bash
ffprobe -v error -show_entries format=duration,size -of json samples/golden.mp4
sha256sum samples/golden.mp4
```

片段包含多名角色、机器人手臂、控制室操作、连续英文对白以及明显的动作变化，足以覆盖 M0 的视觉、对白和时间码契约验证。M0 只验证证据时间线，不把该片段当成最终中文比赛演示素材。
