import cv2
import numpy as np
import json
import logging
import torch
from yomitoku import OCR as YomiTokuOCR
import os
import base64
import io
import flask
from PIL import Image

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask アプリケーション
app = flask.Flask(__name__)

# グローバル変数
ocr_model = None
device = None


def model_fn(model_dir):
    """モデルロード関数"""
    global ocr_model, device

    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # OCRインスタンスの作成
    try:
        ocr_model = YomiTokuOCR(visualize=False, device=device)
        logger.info("OCR model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading OCR model: {str(e)}")
        raise

    return ocr_model


def input_fn(request_body, request_content_type):
    """
    入力データの処理
    - image/jpeg, image/png: バイナリ画像データ
    - application/json: Base64エンコードされた画像データ
    """
    logger.info(f"Received content type: {request_content_type}")

    try:
        if request_content_type == 'application/json':
            # JSONリクエストからBase64エンコードされた画像を取得
            input_data = json.loads(request_body)
            if 'image' in input_data:
                image_data = base64.b64decode(input_data['image'])
                return {'image_data': image_data}
            else:
                return {'error': 'No image field in the JSON request'}

        elif request_content_type.startswith('image/'):
            # 直接バイナリ画像データを受け取る
            return {'image_data': request_body}

        else:
            return {'error': f'Unsupported content type: {request_content_type}'}

    except Exception as e:
        logger.error(f"Error in input processing: {str(e)}")
        return {'error': str(e)}


def perform_ocr(image_data):
    """
    画像データに対してOCR処理を実行する関数
    参考: 提供されたコードをSageMaker用に調整
    """
    try:
        # 画像をOpenCV形式に変換
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("画像をデコードできませんでした")
            return {"error": "Failed to decode image", "words": []}

        # OCR処理
        results, _ = ocr_model(img)

        # 結果を構造化
        json_data = {"words": []}
        if hasattr(results, 'words'):
            for i, word in enumerate(results.words):
                word_dict = {
                    "id": i,
                    "content": word.content,
                    "direction": word.direction,
                    "det_score": float(word.det_score),
                    "rec_score": float(word.rec_score),
                    "points": word.points.tolist() if hasattr(word.points, 'tolist') else word.points
                }
                json_data["words"].append(word_dict)

        logger.info(f"OCR完了: {len(json_data['words'])}単語を検出")
        return json_data

    except Exception as e:
        logger.error(f"OCR処理エラー: {str(e)}")
        return {"error": str(e), "words": []}


def predict_fn(input_data, model):
    """推論関数"""
    # エラーチェック
    if 'error' in input_data:
        return {'error': input_data['error'], 'words': []}

    if 'image_data' not in input_data:
        return {'error': 'No image data available', 'words': []}

    # OCR処理を実行
    return perform_ocr(input_data['image_data'])


def output_fn(prediction, response_content_type):
    """出力データの処理"""
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")


# SageMaker健全性チェック用エンドポイント
@app.route('/ping', methods=['GET'])
def ping():
    """
    SageMakerによって呼び出されるヘルスチェックエンドポイント
    """
    # モデルが読み込まれているかチェック
    health = ocr_model is not None
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


# SageMaker推論用エンドポイント
@app.route('/invocations', methods=['POST'])
def invoke():
    """
    SageMakerによって呼び出される推論エンドポイント
    """
    # リクエストデータ取得
    content_type = flask.request.content_type

    # バイナリデータとして読み込む
    request_body = flask.request.get_data()

    # 入力データ処理
    input_data = input_fn(request_body, content_type)

    # 推論実行
    prediction = predict_fn(input_data, ocr_model)

    # 出力形式に変換
    response = output_fn(prediction, 'application/json')

    return flask.Response(response=response, status=200, mimetype='application/json')


if __name__ == '__main__':
    # モデルロード
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    ocr_model = model_fn(model_dir)

    # サーバー起動（SageMakerが期待するポート）
    app.run(host='0.0.0.0', port=8080)
