# Longform v1 素材与许可

ShotSeek Longform v1 是一个 **36:58.672 的连续单文件开放短片合辑**，用于验证
Production Runtime 的长视频规模、跨作品检索、时间码与证据对齐。它不是单一叙事
长片，因此报告中不得把它描述成“一部 37 分钟电影”。

媒体文件位于被 Git 忽略的
`samples/shotseek_longform_blender_open_movies_v1.mp4`，公开仓库只保存构建脚本、
许可与哈希，不分发约 864 MB 的视频。

## 内容与原片偏移

| 顺序 | 作品 | 合辑区间 | 原片时长 | 许可 |
|---:|---|---|---:|---|
| 1 | Tears of Steel | 00:00:00.000–00:12:14.167 | 12:14.167 | CC BY 3.0 |
| 2 | Sintel | 00:12:14.167–00:27:02.199 | 14:48.032 | CC BY 3.0 |
| 3 | Big Buck Bunny | 00:27:02.199–00:36:58.661 | 09:56.462 | CC BY 3.0 |

合成后的音频重采样使容器尾部比原片时长和多 11 ms，最终时长为 2,218,672 ms。

## 官方来源与署名

### Tears of Steel

- 下载：
  <https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov>
- 项目与许可：<https://mango.blender.org/about/>
- 署名：© Blender Foundation / Blender Institute，完整片尾保留。
- SHA-256：
  `efa9062d9cdb7a338e40ad530dfdf234806743f29ae6a1a136b97ece4e588e8f`

### Sintel

- 下载：<https://download.blender.org/demo/movies/Sintel.2010.720p.mkv>
- 官方许可说明：<https://durian.blender.org/sharing/>
- 署名：© Blender Foundation | durian.blender.org，完整片尾保留。
- SHA-256：
  `60cff51761641626e82eeb4e1c248c471375b2536bb1089f49825b7fb58d8723`
- 官方 MD5 与本地一致：
  `08d1108e0160b847f894acfdbce82305`

### Big Buck Bunny

- 下载：
  <https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_720p_h264.mov>
- Blender 官方视频页与许可：
  <https://video.blender.org/w/pAQiVCgv2CsLg79KKXUoMw>
- 署名：© 2008 Blender Foundation，完整片尾保留。
- SHA-256：
  `45c8bafeb9a53df7f491198d2e71529701bcf1cd51805782089fac1d32869f9b`

## 可复现构建

```bash
python scripts/build_longform_sample.py
```

脚本固定执行以下约束：

- 逐个核验原片 SHA-256；
- 按 Tears of Steel → Sintel → Big Buck Bunny 的顺序拼接；
- 统一为 1280×720、25fps CFR、H.264；
- 音频统一为 48kHz 立体声 AAC 96kbps；
- 优先使用 `h264_nvenc`，失败时明确回退 `libx264`；
- 生成同目录 `.manifest.json`，记录全部偏移、探测结果和输出哈希。

## 构建验收

| 项目 | 结果 |
|---|---:|
| 输出 SHA-256 | `cdb46b849c57b8a70b326f3d9c95f6f72e189bf0d3a56438f54c73b9045c3b89` |
| 文件大小 | 863,775,463 bytes |
| 时长 | 2,218,672 ms |
| 视频 | H.264，1280×720，25/1 CFR |
| 音频 | AAC，48kHz，2 channels |
| 总帧数 | 55,466 |
| 完整视频/音频解码 | PASS |
| 实际编码器 | h264_nvenc |
| 片尾署名 | 三部作品均完整保留 |

该素材只用于开发、评测与比赛演示。若发布含影片画面的演示视频，必须继续保留作品
署名，并在演示说明中列出上述来源与许可。
