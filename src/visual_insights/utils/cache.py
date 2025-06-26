"""
缓存管理模块

提供缓存机制以避免重复计算耗时的操作，如 SHAP 值和 Sobol 指数计算。
"""

import gc
import logging
from typing import Any

logger = logging.getLogger(__name__)

class Cache:
    """
    通用缓存管理类
    """
    def __init__(self, name: str, max_size: int | None = None):
        """
        初始化缓存
        
        参数:
            name: 缓存名称
            max_size: 最大缓存项数量（None 表示无限制）
        """
        self.name = name
        self.max_size = max_size
        self._cache: dict[str, Any] = {}
        self._access_count: dict[str, int] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存项
        
        参数:
            key: 缓存键
            default: 如果键不存在，返回的默认值
            
        返回:
            缓存的值或默认值
        """
        if key in self._cache:
            self._access_count[key] += 1
            logger.debug(f"Cache hit for {self.name}: {key}")
            return self._cache[key]
        logger.debug(f"Cache miss for {self.name}: {key}")
        return default

    def set(self, key: str, value: Any) -> None:
        """
        设置缓存项
        
        参数:
            key: 缓存键
            value: 要缓存的值
        """
        # 如果达到最大大小，删除最少访问的项
        if self.max_size is not None and len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_least_used()

        self._cache[key] = value
        self._access_count[key] = 0
        logger.debug(f"Cache set for {self.name}: {key}")

    def _evict_least_used(self) -> None:
        """删除最少使用的缓存项"""
        if not self._cache:
            return

        # 找到访问次数最少的键
        min_key = min(self._access_count.items(), key=lambda x: x[1])[0]

        # 删除该项
        del self._cache[min_key]
        del self._access_count[min_key]
        logger.debug(f"Cache evicted for {self.name}: {min_key}")

    def clear(self) -> None:
        """清除所有缓存"""
        self._cache.clear()
        self._access_count.clear()
        logger.debug(f"Cache cleared for {self.name}")

    def __contains__(self, key: str) -> bool:
        """检查键是否在缓存中"""
        return key in self._cache

    def __len__(self) -> int:
        """返回缓存项数量"""
        return len(self._cache)


# 创建全局缓存实例
explainer_cache = Cache("explainer", max_size=10)
shap_values_cache = Cache("shap_values", max_size=20)
sobol_indices_cache = Cache("sobol_indices", max_size=10)

def clear_cache(module: str | None = None) -> None:
    """
    清除缓存
    
    参数:
        module: 要清除缓存的模块名称（如 'shap', 'sobol'），None 表示清除所有缓存
    """
    if module is None or module == 'all':
        explainer_cache.clear()
        shap_values_cache.clear()
        sobol_indices_cache.clear()
        gc.collect()  # 强制垃圾回收
        logger.info("All caches cleared")
    elif module == 'shap':
        explainer_cache.clear()
        shap_values_cache.clear()
        gc.collect()
        logger.info("SHAP caches cleared")
    elif module == 'sobol':
        sobol_indices_cache.clear()
        gc.collect()
        logger.info("Sobol caches cleared")
    else:
        logger.warning(f"Unknown module: {module}, no cache cleared")
