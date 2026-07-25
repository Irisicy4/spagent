# Detection Encoding 三种策略

后端统一用 GroundingDINO，三种策略的区别在于**把检测结果以什么形式喂给 LLM**。

---

## Encoding 1：`image-only`（仅图像）

工具调用后，把**带 bbox 标注的图像**作为额外图像附加到下一轮 continuation prompt 里，不附加任何文字描述。

```python
# tool 返回
{
    "output_path": "<annotated image with bboxes drawn>",
    "vis_path":    "<same>",
    "boxes":       [[x1, y1, x2, y2], ...],   # normalized [0,1]
    "labels":      ["label1", ...],
}
# LLM 收到：原始图 + 标注图（多了一张图）
```

**实现：** `spagent/tools/detection_tool.py` → `ObjectDetectionTool`

---

## Encoding 2：`image-and-text`（图像 + 文本）

工具返回**标注图像 + JSON 文本描述**，LLM 同时收到两份信息（视觉冗余）。

```python
# tool 返回
{
    "output_path":  "<annotated image>",
    "description":  'Detected objects: [{"label": "person", "bbox": [0.07, 0.25, 0.48, 1.0]}, ...]',
    "detections":   [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...],
}
# LLM 收到：原始图 + 标注图 + text description 注入 continuation prompt
```

**实现：** `spagent/external_experts/GroundingDINO/grounding_dino_image_and_text_tool.py` → `GroundingDINOImageAndTextTool`

---

## Encoding 3：`text-only`（仅文本）

工具**不返回任何图像**，只返回 JSON 格式的检测结果文本。

```python
# tool 返回
{
    "detections": [{"label": "baseball bat", "bbox": [0.27, 0.70, 0.51, 0.83]}, ...],
    "result":     '[{"label": "baseball bat", "bbox": [0.27, 0.70, 0.51, 0.83]}, ...]'
}
# LLM 收到：原始图 + text result（无额外图像）
```

**实现：** `spagent/external_experts/GroundingDINO_text_only/grounding_dino_text_only_tool.py` → `GroundingDINOTextOnlyTool`

---

## 对比总结

| 维度 | image-only | image-and-text | text-only |
|------|-----------|----------------|-----------|
| 给 LLM 的图像 | 原图 + 标注图 | 原图 + 标注图 | 仅原图 |
| 给 LLM 的文本 | 无 bbox 文字 | JSON bbox 列表 | JSON bbox 列表 |
| tool name | `detect_objects_tool` | `detect_objects_tool` | `detect_objects_text_only` |
| bbox 格式 | normalized xyxy [0,1] | normalized xyxy [0,1] | normalized xyxy [0,1] |

---

## 框架集成方式

`spagent/core/spagent.py` 的 `solve()` 方法统一处理两个字段：

- **`output_path` / `vis_path`**（L296–301）：字段存在且文件存在时，自动把该图像追加到下轮 continuation prompt 的图像列表。
- **`description`**（L327–328）：字段存在时，把其内容注入为文字提示。

三种 encoding 的差异完全体现在这两个字段上，框架层无需修改。
