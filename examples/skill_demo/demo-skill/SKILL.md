---
name: demo-skill
description: 一个演示技能，用于展示 Skills 模块的基本功能
author: gearlink
version: 1.0
---

# 示例技能指令

当用户请求使用 demo-skill 时，请执行以下步骤：
1. 确认用户意图。
2. 告知用户当前日期和时间（可以使用 get_current_time 工具获取）。
3. 根据时间返回一句问候语（如“上午好”、“下午好”）。

如果用户问“现在几点了？”，请优先直接使用 get_current_time 工具。