import requests

class Policy_client:
    def __init__(self, base_url):
        self.base_url = base_url

    def __call__(self, prompt):
        # 使用 requests 库发送 POST 请求
        response = requests.post(
            url=f"{self.base_url}/predict",
            json={"prompt": prompt},  # 假设 API 接受 JSON 格式
        )
        # 检查响应状态码
        if response.status_code != 200:
            raise Exception(f"API 请求失败，状态码: {response.status_code}, 内容: {response.text}")
        
        # 解析返回的 JSON 数据
        result = response.json()
        return result['prediction']  # 假设返回的 JSON 中有 'prediction' 字段