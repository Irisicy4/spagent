from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import DepthEstimationTool, SegmentationTool

# Create model and tools
model = GPTModel(model_name="gpt-4o-mini")
tools = [
    DepthEstimationTool(use_mock=False, server_url="http://localhost:20019"),
    SegmentationTool(use_mock=False, server_url="http://localhost:20020"),
]
# Create agent
agent = SPAgent(model=model, tools=tools)

# Solve problem
result = agent.solve_problem("./apple.jpg", "how many apples are there in the image?")
print(result['answer'])