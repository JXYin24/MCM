import os
import torch
import logging
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import requests
from io import BytesIO
from functools import lru_cache
from datetime import datetime
import json
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
import threading

# 创建模型访问锁
model_lock = threading.Lock()

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(title="MCM异常值检测API", description="基于MCM模型的在线异常值检测服务", version="1.0.0")

# 初始化限流
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute", "10/second"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 初始化Prometheus监控
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# 全局变量存储模型和配置
model = None
model_config = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 配置验证模型
class ModelConfig(BaseModel):
    data_dim: int = Field(..., description="输入数据的维度")
    hidden_dim: int = Field(..., description="隐藏层维度")
    z_dim: int = Field(..., description="潜在空间维度")
    mask_num: int = Field(..., description="掩码数量")
    en_nlayers: int = Field(..., description="编码器层数")
    de_nlayers: int = Field(..., description="解码器层数")
    mask_nlayers: int = Field(..., description="掩码网络层数")

# 输入数据验证模型
class AnomalyDetectionRequest(BaseModel):
    data: List[List[float]] = Field(..., description="输入数据，每行代表一个样本")
    threshold: Optional[float] = Field(None, description="异常判定阈值")

# 输出结果模型
class AnomalyDetectionResponse(BaseModel):
    request_id: str = Field(..., description="请求ID")
    timestamp: str = Field(..., description="处理时间戳")
    results: List[dict] = Field(..., description="检测结果列表")
    summary: dict = Field(..., description="检测结果摘要")

# 从本地文件加载模型
@app.post("/load_model/local")
def load_model_local(model_path: str, config: ModelConfig):
    global model, model_config
    
    try:
        # 加载模型
        model = torch.load(model_path, map_location=device)
        model.eval()
        
        # 保存配置
        model_config = config.dict()
        
        logger.info(f"成功从本地文件加载模型: {model_path}")
        return JSONResponse(content={"message": f"成功加载模型: {model_path}", "config": model_config})
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")

# 从远程URL加载模型
@app.post("/load_model/remote")
def load_model_remote(model_url: str, config: ModelConfig):
    global model, model_config
    
    try:
        # 下载模型文件
        response = requests.get(model_url)
        response.raise_for_status()
        
        # 加载模型
        model = torch.load(BytesIO(response.content), map_location=device)
        model.eval()
        
        # 保存配置
        model_config = config.dict()
        
        logger.info(f"成功从远程URL加载模型: {model_url}")
        return JSONResponse(content={"message": f"成功加载模型: {model_url}", "config": model_config})
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")

# 健康检查端点
@app.get("/health")
def health_check():
    return JSONResponse(content={"status": "健康", "model_loaded": model is not None})

# 获取模型配置信息
@app.get("/model_config")
def get_model_config():
    if model_config is None:
        raise HTTPException(status_code=400, detail="模型未加载，请先调用加载模型接口")
    return JSONResponse(content=model_config)

# 异常检测端点
@app.post("/detect_anomalies", response_model=AnomalyDetectionResponse)
def detect_anomalies(request: AnomalyDetectionRequest, req: Request):
    global model, model_config
    
    # 检查模型是否已加载
    if model is None:
        raise HTTPException(status_code=400, detail="模型未加载，请先调用加载模型接口")
    
    try:
        # 生成请求ID
        request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        # 记录请求开始时间
        start_time = datetime.now()
        
        # 请求校验：验证输入数据的完整性、格式正确性及合法性
        data = np.array(request.data)
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="输入数据不能为空")
        
        # 特征对齐：确保输入数据特征与模型训练时使用的特征空间一致
        if data.shape[1] != model_config['data_dim']:
            raise HTTPException(status_code=400, detail=f"输入数据维度不匹配， expected: {model_config['data_dim']}, actual: {data.shape[1]}")
        
        # 模型推理：使用加载的MCM模型对输入数据执行推理计算
        with torch.no_grad():
            with model_lock:
                x_input = torch.tensor(data, dtype=torch.float32).to(device)
                x_pred, z, masks = model(x_input)
                
                # 计算MSE分数
                x_input = x_input.unsqueeze(1).repeat(1, model_config['mask_num'], 1)
                sub_result = x_pred - x_input
                mse = torch.norm(sub_result, p=2, dim=2)
                mse_score = torch.mean(mse, dim=1, keepdim=True)
                
                # 将结果转换为numpy数组
                scores = mse_score.cpu().numpy().flatten()
        
        # 分数与阈值判定：基于模型输出分数与预设阈值进行异常判定
        if request.threshold is None:
            # 如果没有提供阈值，使用默认阈值（可以根据实际情况调整）
            threshold = np.percentile(scores, 95)
        else:
            threshold = request.threshold
        
        # 判定异常
        anomalies = scores > threshold
        
        # 记录请求结束时间
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # 审计落库/日志：将请求详情、推理结果及判定结论进行持久化存储与日志记录
        log_data = {
            "request_id": request_id,
            "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "client_ip": req.client.host,
            "input_shape": data.shape,
            "threshold": threshold,
            "scores": scores.tolist(),
            "anomalies": anomalies.tolist(),
            "processing_time": processing_time
        }
        
        # 保存到日志文件
        with open("detection_log.jsonl", "a") as f:
            json.dump(log_data, f)
            f.write("\n")
        
        # 构建响应结果
        results = []
        for i, (score, is_anomaly) in enumerate(zip(scores, anomalies)):
            results.append({
                "sample_id": i + 1,
                "score": float(score),
                "is_anomaly": bool(is_anomaly)
            })
        
        # 构建摘要
        summary = {
            "total_samples": len(data),
            "anomaly_count": int(np.sum(anomalies)),
            "anomaly_rate": float(np.mean(anomalies)),
            "threshold": float(threshold),
            "processing_time": float(processing_time)
        }
        
        # 返回JSON响应
        return AnomalyDetectionResponse(
            request_id=request_id,
            timestamp=start_time.strftime("%Y-%m-%d %H:%M:%S"),
            results=results,
            summary=summary
        )
    except Exception as e:
        logger.error(f"异常检测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"异常检测失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)