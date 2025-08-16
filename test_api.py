import requests
import json

# Test the local API
API_URL = "http://localhost:5000"


def test_api():
    print("🧪 Testing Local API...")

    # Test 1: Check if API is running
    try:
        response = requests.get(f"{API_URL}/")
        print(f"✅ API Status: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("❌ API is not running! Start app.py first.")
        return

    # Test 2: Single prediction
    test_data = {"text": "I hate you so much"}
    try:
        response = requests.post(f"{API_URL}/predict", json=test_data)
        result = response.json()
        print(f"✅ Single prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
    except Exception as e:
        print(f"❌ Single prediction failed: {e}")

    # Test 3: Batch prediction
    batch_data = {"texts": ["Kill yourself now",
    "You disgusting freak",
                            "أنت قمامة عديمة القيمة",
                            "مت أيها الخاسر المثير للشفقة",
                            "أنت مقزز ومثير للاشمئزاز",
                            "Öl sen acınası kaybeden",
                            "Sen iğrenç bir yaratıksın",
                            "Keşke ölsen",
                            "Sus aptal",
                            "I appreciate your hard work",
                            "طقس جميل اليوم",
                            "Harika yardımınız için teşekkürler",
                            ]}
    try:
        response = requests.post(f"{API_URL}/batch_predict", json=batch_data)
        results = response.json()['results']
        print(f"✅ Batch prediction: {len(results)} texts processed")
        for result in results:
            print(f"   '{result['text']}' -> {result['prediction']}")
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")


if __name__ == "__main__":
    test_api()