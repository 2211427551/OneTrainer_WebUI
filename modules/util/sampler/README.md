# MeanCache 採樣加速

## 概述

MeanCache 是一個針對 Flow Matching 模型（如 Z-Image）的訓練無需（training-free）推理加速技術。

## 工作原理

### 核心算法

1. **JVP 速度修正**
   - 使用 Jacobian-Vector Product (JVP) 近似計算速度變化
   - 公式：`JVP_{r→t} ≈ (v_t - v_r) / (t - r)`
   - 使用平均速度代替瞬時速度

2. **智能跳步**
   - 當速度場穩定時跳過某些步驟
   - 使用快取的 JVP 修正速度進行預測
   - 動態調整跳步率以達到目標加速比

3. **預設檔案**
   - **Quality** (1.4x): 高品質，較少跳步（30% 跳步率）
   - **Balanced** (1.67x): 平衡品質與速度（40% 跳步率）
   - **Speed** (1.8x): 優先速度（45% 跳步率）
   - **Turbo** (2.0x): 最大加速（50% 跳步率）

## 使用方法

### UI 設定

1. 打開 **Sampling** 標籤或採樣視窗
2. 啟用 **MeanCache** 開關
3. 選擇預設檔案（Quality / Balanced / Speed / Turbo）

### 適用模型

- ✅ Flow Matching 模型（Z-Image、SD3 等）
- ❌ DDPM/DDIM 模型（Stable Diffusion 1.x/2.x/XL）

## 實現狀態

### ✅ 已完成
1. 配置系統整合（`SampleConfig`）
2. UI 控制項（開關 + 預設選擇）
3. MeanCache 包裝器框架（`meancache_wrapper.py`）

### ⚠️ 待完成
1. **與 Sampler 整合**
   - 需要在實際的採樣管線中應用 MeanCacheWrapper
   - 修改模型前向傳播邏輯

2. **完整的 L_K 穩定性度量**
   - 目前使用簡化的啟發式方法
   - 應實現完整的穩定性偏差計算：
     `L_K = ||v_current - (v_prev + dt · JVP)|| / ||v_current||`

3. **PSSP 調度演算法**
   - 動態規劃尋找最佳計算路徑
   - 目前僅使用固定跳步策略

4. **測試與驗證**
   - 需要實際測試加速效果
   - 驗證圖像品質是否受影響

## 技術細節

### 狀態管理
- `prev_velocity`: 上一步的速度
- `prev_timestep`: 上一步的時間步
- `prev_jvp`: 快取的 JVP 值

### 跳步決策
```python
if self._should_skip(timestep):
    # 使用 JVP 修正的速度
    predicted_velocity = prev_velocity + dt * prev_jvp
    return predicted_velocity
else:
    # 實際計算速度並更新 JVP
    velocity = model(...)
    prev_jvp = (velocity - prev_velocity) / dt
    return velocity
```

## 參考資料

- [MeanCache 論文](https://unicomai.github.io/MeanCache/)
- [ComfyUI 實現](https://github.com/facok/comfyui-meancache-z)

## 已知限制

1. **僅支援 Flow Matching 模型**
   - 不適用於 DDPM/DDIM 等傳統擴散模型
   
2. **簡化實現**
   - 當前版本使用啟發式跳步策略
   - 完整版需要更複雜的穩定性分析

3. **需要進一步整合**
   - 目前僅建立框架
   - 需要與實際採樣器整合才能使用

## 後續工作

1. 找到 OneTrainer 中實際的採樣器代碼
2. 將 MeanCacheWrapper 整合到採樣管線
3. 實現完整的 L_K 穩定性度量
4. 添加 PSSP 調度演算法
5. 進行測試並調整參數
