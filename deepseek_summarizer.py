#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import sys
import argparse
import datetime
import logging

import pytz
from openai import OpenAI

# 设置日志
log_format = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
log_datefmt = '%Y-%m-%d %H:%M:%S %z'

# 配置根日志记录器
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    datefmt=log_datefmt,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("deepseek_summarizer")
logger.setLevel(logging.INFO)

# 常量定义
ARTICLES_DIR = "articles"
DAILYBRIEF_DIR = "dailybrief"
DEFAULT_MODEL = "deepseek-chat"
DEEPSEEK_API_URL = "https://api.deepseek.com"

def ensure_dir_exists(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"创建目录: {directory}")

def get_eastern_time():
    """获取当前美东时间"""
    eastern = pytz.timezone('US/Eastern')
    return datetime.datetime.now(eastern)

# 更新后的提示词模板
DEFAULT_PROMPT = """作为资深投资顾问以及新闻编辑，请将以下JSON格式的新闻条目整合为每日简报。
## 要求
1. 首段总结当日核心事件
2. 每个新闻单独段落，包含：
   - 关键事实
   - 影响分析
   - 相关背景
3. 结尾添加投资决策建议及风险提示
4. 使用Markdown格式，注意排版美观、专业、易于阅读
5. 对所有内容生成双语版本（中/英），以便对比阅读。注意格式美观、专业、易于阅读

## 注意
输出仅返回markdown内容（不要包含```markdown等符号），请勿返回任何其他内容。你的内容将直接用于展示。
"""

def load_articles(date_str=None):
    """加载指定日期的文章"""
    try:
        if not date_str:
            now = get_eastern_time()
            date_str = now.strftime("%Y%m%d")
        
        filepath = os.path.join(ARTICLES_DIR, f"{date_str}.json")
        
        if not os.path.exists(filepath):
            logger.error(f"文件不存在: {filepath}")
            return None
        
        with open(filepath, "r", encoding="utf-8") as f:
            articles = json.load(f)
            logger.info(f"从 {filepath} 加载了 {len(articles)} 篇文章")
            return articles
    except Exception as e:
        logger.error(f"加载文章失败: {str(e)}")
        return None

def call_deepseek_api(api_key=None, prompt=None, articles=None):
    """调用DeepSeek API生成摘要"""
    try:
        # 获取API密钥
        if api_key is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                logger.error("未设置DEEPSEEK_API_KEY环境变量")
                return None
        
        # 创建OpenAI客户端并配置DeepSeek的API端点
        client = OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_API_URL
        )
        
        # 构造系统与用户消息
        messages = [
            {
                "role": "user",
                "content": f"{prompt}\n\n输入数据:\n{json.dumps(articles, ensure_ascii=False, indent=2)}"
            },
        ]

        # 调用DeepSeek API
        response = client.chat.completions.create(
            model=os.environ.get('SUMMARY_MODEL', DEFAULT_MODEL),  # 固定使用DeepSeek聊天模型
            messages=messages,
            temperature=0.6,        # 控制生成随机性
            max_tokens=4096,        # 限制生成长度
            stream=False            # 同步请求
        )

        
        # 提取并返回生成的文本
        if response.choices and len(response.choices) > 0:
            text = response.choices[0].message.content
            logger.info("成功生成摘要")
            return text
        else:
            logger.error("API响应中未找到有效的choices")
            return None
            
    except Exception as e:
        logger.error(f"调用DeepSeek API失败: {str(e)}")
        return None


def save_daily_brief(content, date_str=None):
    """保存每日简报"""
    try:
        if not date_str:
            now = get_eastern_time()
            date_str = now.strftime("%Y%m%d")
        
        ensure_dir_exists(DAILYBRIEF_DIR)
        filepath = os.path.join(DAILYBRIEF_DIR, f"{date_str}.md")
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"简报已保存到 {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"保存简报失败: {str(e)}")
        return None

def generate_daily_brief(api_key=None, date_str=None):
    """生成每日简报"""
    prompt = DEFAULT_PROMPT
    
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            logger.error("需要DEEPSEEK_API_KEY")
            return False
    
    articles = load_articles(date_str)
    if not articles:
        return False
    
    logger.info("调用DeepSeek API...")
    summary = call_deepseek_api(api_key, prompt, articles)
    if not summary:
        return False
    
    return save_daily_brief(summary, date_str) is not None

def main():
    parser = argparse.ArgumentParser(description="DeepSeek每日新闻简报生成器")
    parser.add_argument("--api-key", help="DeepSeek API密钥")
    parser.add_argument("--date", help="指定日期 (YYYYMMDD)")
    parser.add_argument("--model", help="DeepSeek模型名称")
    
    args = parser.parse_args()
    
    if args.model:
        os.environ["DEEPSEEK_MODEL"] = args.model
    
    if args.api_key:
        os.environ["DEEPSEEK_API_KEY"] = args.api_key
    
    ensure_dir_exists(DAILYBRIEF_DIR)
    
    success = generate_daily_brief(date_str=args.date)
    
    if success:
        logger.info("简报生成成功")
    else:
        logger.error("简报生成失败")

if __name__ == "__main__":
    main()
