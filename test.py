# from inference_sdk import InferenceHTTPClient

# client = InferenceHTTPClient(
#     api_url="https://serverless.roboflow.com",
#     api_key="3SvrFkAklozlftLakk1Z",
# ) 

# result = client.run_workflow(
#     workspace_name="sarawans-workspace",
#     workflow_id="custom-workflow",  # confirm this is correct
#     images={"image": "/images/7119-175.jpg"}, 
#     use_cache=False,
# )

# print(result)

import base64
from io import BytesIO
from PIL import Image
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="3SvrFkAklozlftLakk1Z",
)

image = Image.open("images/7119-175.jpg").convert("RGB")
buffered = BytesIO()
image.save(buffered, format="JPEG", quality=100)
img_b64 = base64.b64encode(buffered.getvalue()).decode("ascii")

result = client.run_workflow(
    workspace_name="sarawans-workspace",
    workflow_id="custom-workflow",
    images={"image": img_b64},
    use_cache=False,
)
print(result)
import json
print(json.dumps(result, indent=2, default=str))

