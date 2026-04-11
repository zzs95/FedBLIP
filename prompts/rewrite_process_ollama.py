from ollama import Client

class LLM_pipline():
    def __init__(self, model_id="gpt-oss:20b", gpu='0'):
        gpu_to_host = {str(i): f"http://localhost:{11435 + i}" for i in range(8)}  # 0-7
        if gpu not in gpu_to_host:
            raise ValueError(f"Invalid gpu='{gpu}'. Expected one of {list(gpu_to_host.keys())}")

        self.client = Client(
            host=gpu_to_host[gpu],
            headers={"x-some-header": "some-value"},
        )
        self.model_name = model_id
        

    def forward(self, content_text='',):
        response = self.client.generate(model=self.model_name, prompt=content_text)
        generated_text0 = response['response']
        return generated_text0