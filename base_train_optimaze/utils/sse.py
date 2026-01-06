import json
import zipfile
import os
import sys
import glob
from pathlib import Path
import numpy as np

def sse_print(event: str, data: dict) -> str:
    """
    SSE 打印
    :param event: 事件名称
    :param data: 事件数据（字典或能被 json 序列化的对象）
    :return: SSE 格式字符串
    """
    # 将数据转成 JSON 字符串
    json_str = json.dumps(data, ensure_ascii=False, default=lambda obj: obj.item() if isinstance(obj, np.generic) else obj)
    
    # 按 SSE 协议格式拼接
    message = f"event: {event}\n" \
              f"data: {json_str}\n"
    print(message)


def sse_input_path_validated(args):
    try:
        if os.path.exists(args.input_path):
            event = "input_path_validated"
            data = {
                "status": "success",
                "message": "Input path is valid and complete.",
                "file_name": args.input_path
            }
            sse_print(event, data)
            
            try:
                if os.path.exists(f'{args.input_path}/data'):
                    event = "input_data_validated"
                    data = {
                        "status": "success",
                        "message": "Input data file is valid and complete.",
                        "file_name": glob.glob(os.path.join(f'{args.input_path}/data', '*/'))[0]
                    }
                    sse_print(event, data)
                else:
                    raise ValueError('Input data file not found.')
            except Exception as e:
                event = "input_data_validated"
                data = {
                    "status": "failure",
                    "message": f"{e}"
                }
                sse_print(event, data)
                
            try:
                if os.path.exists(f'{args.input_path}/model'):
                    event = "input_model_validated"
                    data = {
                        "status": "success",
                        "message": "Input model file is valid and complete.",
                        "file_name": glob.glob(os.path.join(f'{args.input_path}/model', '*/'))[0]
                    }
                    sse_print(event, data)
                else:
                    raise ValueError('Input model file not found.')
            except Exception as e:
                event = "input_model_validated"
                data = {
                    "status": "failure",
                    "message": f"{e}"
                }
                sse_print(event, data)
        else:
            raise ValueError('Input path not found.')
    except Exception as e:
        event = "input_path_validated"
        data = {
            "status": "failure",
            "message": f"{e}"
        }
        sse_print(event, data)
        
    
def sse_output_path_validated(args):
    try:
        if os.path.exists(args.output_path):
            event = "output_path_validated"
            data = {
                "status": "success",
                "message": "Output path is valid and complete.",
                "file_name": args.output_path
            }
            sse_print(event, data)
        else:
            raise ValueError('Output path not found.')
    except Exception as e:
        event = "output_path_validated"
        data = {
            "status": "failure",
            "message": f"{e}"
        }
        sse_print(event, data)
            
        